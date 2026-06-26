"""Orchestrate: inventory → proposal → strategy → execute → narrative → report."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .cache import Cache
from .config import AppSettings
from .discovery import inventory, scan
from .execute.runner import Runner
from .integrations import oncat
from .llm import LLMClient
from .models import (
    ExperimentContext,
    ProposalInfo,
    ReportModel,
)
from .narrative import synthesize
from .proposal import summarize
from .report import assemble, compile as texcompile, tables
from .strategy import engine

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    out_dir: Path
    pdfs: list[Path]
    tex_files: list[Path]
    strategy: object
    context: ExperimentContext


def _log_step(step, name, args):
    short = {k: (v if not isinstance(v, (list, dict)) else f"<{len(v)} items>")
             for k, v in args.items()}
    logger.info("  [strategy step %d] %s(%s)", step, name, short)


def run_report(
    target: str,
    out_dir: str | Path,
    *,
    no_llm: bool = False,
    no_proposal: bool = False,
    strategy_only: bool = False,
    refresh: bool = False,
    sasfit: bool = True,
    max_llm_steps: Optional[int] = None,
) -> RunResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # --refresh busts ALL caches (LLM, strategy, sasfit, proposal, ONCat): every
    # read misses and recomputes, then rewrites the cache for next time.
    cache = Cache(out_dir / ".reportit_cache", enabled=True, bust=refresh)

    settings = AppSettings.load()
    steps = max_llm_steps or settings.max_llm_steps

    llm: LLMClient | None = None
    if not no_llm:
        if settings.llm.is_configured:
            llm = LLMClient(settings.llm, cache=cache)
        else:
            logger.warning("No OPENROUTER_API_KEY found — running in --no-llm mode.")

    # 1) inventory
    logger.info("Building inventory for %s ...", target)
    inv = inventory.build(target)
    ctx = ExperimentContext(ipts=inv.ipts, shared_dir=inv.shared_dir, inventory=inv)

    # 2) scan datasets across all candidate output dirs (variants)
    datasets = []
    for odir in inv.output_dirs:
        datasets.extend(scan.scan_dir(odir, odir.name))
    ctx.datasets = datasets
    logger.info("Discovered %d datasets across %d variant dir(s).",
                len(datasets), len(inv.output_dirs))
    if not datasets:
        ctx.degraded.append("No reduced datasets found.")

    # 3) ONCat catalog (cached) → fill titles
    catalog = oncat.fetch_catalog_cached(inv.ipts, cache, refresh=refresh) if inv.ipts else None
    ctx.catalog = catalog
    if catalog is None or getattr(catalog, "empty", True):
        ctx.degraded.append("ONCat catalog unavailable — titles inferred from filenames.")
    else:
        _fill_titles(datasets, catalog)

    # 4) proposal
    proposal = ProposalInfo()
    if not no_proposal and inv.proposal_pdfs:
        logger.info("Reading proposal: %s", ", ".join(p.name for p in inv.proposal_pdfs))
        proposal = summarize.summarize(inv.proposal_pdfs, llm)
        if not proposal.available:
            ctx.degraded.append("Proposal PDF present but no extractable text.")
    elif not inv.proposal_pdfs:
        ctx.degraded.append("No proposal document found.")
    ctx.proposal = proposal

    # 5) strategy (agentic LLM or deterministic)
    logger.info("Deriving analysis strategy (%s)...", "LLM" if llm else "deterministic")
    strategy = engine.derive_strategy(inv, datasets, proposal, llm, catalog=catalog,
                                      max_steps=steps, on_step=_log_step)
    ctx.degraded.extend(strategy.caveats)

    if strategy_only:
        _print_strategy(strategy)
        return RunResult(out_dir, [], [], strategy, ctx)

    # 6) execute → group reports
    logger.info("Executing strategy: %d group(s)...", len(strategy.groups))
    runner = Runner(datasets, out_dir / "figures")
    group_reports = runner.run(strategy)

    # 7) per-group observations (grounded in the actual plot + experiment context)
    obs_context = strategy.experiment_summary
    if proposal and proposal.science_goals:
        obs_context += " Goals: " + "; ".join(proposal.science_goals)
    for gr in group_reports:
        gr.observations = synthesize.observe_group(gr, llm, context=obs_context)

    # 7b) agentic model-based fitting (sasmodels) — opt-in
    sas_outcomes = []
    if sasfit and llm is not None:
        from .analysis import sas_agent
        fig_dir = out_dir / "figures"
        name_to_ds: dict[str, list] = {}
        for d in datasets:
            name_to_ds.setdefault(d.output_name, []).append(d)
        variants = set(strategy.variant_decision.variants_used or [])
        # resolve members up-front so we know how many groups will actually be fit
        pending = []
        for g in strategy.groups:
            members = []
            for nm in g.members:
                cands = [d for d in name_to_ds.get(nm, [])
                         if not variants or d.variant in variants] or name_to_ds.get(nm, [])
                if cands:
                    members.append(cands[0])
            if members:
                pending.append((g, members))

        total = len(pending)
        logger.info("sasfit: model-based fitting for %d group(s)...", total)
        for idx, (g, members) in enumerate(pending, 1):
            logger.info("sasfit: [%d/%d] fitting group %s (%d members) ...",
                        idx, total, g.group_id, len(members))
            try:
                outcome = sas_agent.run_group_fit(
                    g, members, llm, fig_dir, strategy.experiment_summary)
                sas_outcomes.append(outcome)
                model = outcome.best.model_name if outcome.best else "none"
                logger.info("sasfit: [%d/%d] %s -> %s (%s)", idx, total, g.group_id,
                            model, "accepted" if outcome.success else "no satisfactory fit")
            except Exception as e:  # noqa: BLE001
                logger.warning("sasfit: [%d/%d] failed for %s: %s", idx, total, g.group_id, e)

    # 8) global narrative + hypothesis checks
    overview, discussion, checks = synthesize.global_narrative(
        strategy, group_reports, proposal, llm)

    # 9) build report model
    caveats = list(ctx.degraded) + list(strategy.caveats)
    if strategy.variant_decision.rationale:
        caveats.append("Variant choice: " + strategy.variant_decision.rationale)
    appendix = [t for t in (
        tables.build_reduction_table(datasets, strategy),
        tables.build_catalog_table(catalog),
    ) if t is not None]
    model = ReportModel(
        context=ctx,
        title=_title(ctx, proposal),
        overview=overview or strategy.experiment_summary,
        catalog_table=tables.build_sample_summary(datasets, strategy, proposal),
        appendix_tables=appendix,
        group_reports=group_reports,
        sas_fits=sas_outcomes,
        hypothesis_checks=checks,
        discussion=discussion,
        caveats=_dedupe(caveats),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        model_name=(settings.llm.model if llm else "deterministic"),
    )

    # 10) assemble + compile two PDFs
    pdfs, tex_files = [], []
    for mode in ("comprehensive", "summary"):
        tex = assemble.write_tex(model, out_dir, mode)
        tex_files.append(tex)
        pdf = texcompile.compile_pdf(tex)
        if pdf:
            pdfs.append(pdf)
    logger.info("Wrote %d PDF(s), %d tex file(s) to %s", len(pdfs), len(tex_files), out_dir)

    return RunResult(out_dir, pdfs, tex_files, strategy, ctx)


# --------------------------------------------------------------------------- #
def _fill_titles(datasets, catalog) -> None:
    by_run = {}
    try:
        for _, row in catalog.iterrows():
            by_run[int(row["run_number"])] = str(row.get("title", ""))
    except Exception:  # noqa: BLE001
        return
    for d in datasets:
        run = d.meta.sample_run if d.meta else None
        if run:
            try:
                d.oncat_title = by_run.get(int(run))
            except (ValueError, TypeError):
                pass


def _title(ctx, proposal) -> str:
    if proposal and proposal.title:
        return f"EQSANS IPTS-{ctx.ipts}: {proposal.title}"
    return f"EQSANS Experiment Report — IPTS-{ctx.ipts}"


def _dedupe(items) -> list[str]:
    out, seen = [], set()
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _print_strategy(strategy) -> None:
    import json
    from dataclasses import asdict
    print(json.dumps(asdict(strategy), indent=2, default=str))
