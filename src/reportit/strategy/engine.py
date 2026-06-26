"""The agentic strategy loop: inventory + proposal → LLM probes → AnalysisStrategy.

Also provides a deterministic fallback strategy for --no-llm runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..llm import LLMClient
from ..llm.tools import FINALIZE_TOOL, all_tools
from ..models import (
    AnalysisStrategy,
    Dataset,
    FitPlan,
    FolderInventory,
    ProposalInfo,
    StrategyGroup,
    VariantDecision,
)
from .probes import Probes

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert neutron small-angle scattering (SANS) scientist and \
data analyst at the SNS EQSANS beamline. You are given an inventory of an \
experiment's shared data folder and a summary of its proposal. Your job is to \
FIGURE OUT what the experiment is about and DEVISE A STRATEGY for an automated \
report.

Work like a detective: use the provided read-only tools to inspect the folder \
before deciding. Typical useful steps:
  - read NOTE.md / README to learn what the experimenter actually did,
  - list_datasets to see all reduced outputs and their parsed base/temp/config,
  - parse a few reduction .json files to learn run numbers, thickness, mask, scale,
  - oncat_titles on sample run numbers to learn what samples really are,
  - sample_curve on representative datasets to SEE the scattering shape,
  - head_file to confirm data columns.

Key decisions you must ground in evidence, not assumptions:
  - If there are multiple output directories (e.g. output vs output_mask4 with \
different detector masks), decide which to use, or whether to compare them, and \
WHY (read NOTE.md and compare). STRONGLY prefer a variant that HAS combined/merged \
extended-Q profiles (see the per-output-dir coverage in the inventory) — a variant \
with only per-config 1D files and no merged data gives much worse plots and fits, \
so do not pick it when a merged-bearing variant exists. Do not compare a \
merged-bearing variant against one that lacks merged data.
  - Decide curve_source: do combined/stitched 1D profiles exist (see the \
inventory's combined-files list — names vary: merged_*, *_stitched, etc.)? If so, \
prefer 'combined' (extended Q from joining configurations). If there are NONE, use \
'individual' (per-configuration *_Iq.dat). Don't assume merged files exist — check.
  - Exclude calibration standards (e.g. porsil) from science groups.
  - Group datasets into meaningful comparisons: temperature series, concentration \
series, config sets, etc. Order them sensibly.
  - For each group decide whether a quantitative model fit is sensible, and WHICH \
one, by actually looking at the curve shape via sample_curve. Guidance:
      * Guinier (Rg) ONLY when there is a clear low-Q plateau that bends into a \
knee — i.e. compact, finite-size particles. Do NOT default to Guinier; a curve \
that keeps rising toward low Q or is a featureless power law is NOT a Guinier case.
      * correlation (Ornstein-Zernike correlation length xi) when the curve has a \
low-Q plateau rolling into a power-law decay — typical of polymer/solution \
scattering. This is often the right choice for single-chain or network solutions.
      * porod / powerlaw when the curve is dominated by a power-law slope \
(interfaces, networks, mass/surface fractals).
      * Note that the 1-2 lowest-Q points are frequently beam-stop/mask artifacts \
(outliers); the tool already trims them, so do not let them drive your choice.
    Set q_min/q_max to the region where the chosen model actually applies.

Be thorough — call as many tools as you need. When confident, call \
`finalize_strategy` exactly once with a complete, well-justified strategy."""


def _context_message(inv: FolderInventory, proposal: ProposalInfo) -> str:
    lines = ["=== FOLDER INVENTORY ===", inv.as_text(), "", "=== PROPOSAL SUMMARY ==="]
    if proposal and proposal.available:
        lines.append(f"Title: {proposal.title}")
        lines.append(f"PI: {proposal.pi}")
        lines.append(f"Summary: {proposal.abstract_summary}")
        if proposal.science_goals:
            lines.append("Goals: " + "; ".join(proposal.science_goals))
        if proposal.hypotheses:
            lines.append("Hypotheses:")
            for h in proposal.hypotheses:
                lines.append(f"  - {h.text} (look for: {h.expected_signature})")
        if proposal.sample_descriptions:
            lines.append("Sample descriptions: " + str(proposal.sample_descriptions))
    else:
        lines.append("(No usable proposal text — rely on data + ONCat titles.)")
    lines.append("\nNow investigate using the tools, then call finalize_strategy.")
    return "\n".join(lines)


def derive_strategy(
    inv: FolderInventory,
    datasets: list[Dataset],
    proposal: ProposalInfo,
    llm: LLMClient | None,
    catalog=None,
    max_steps: int = 30,
    on_step=None,
) -> AnalysisStrategy:
    if llm is None:
        return deterministic_strategy(datasets, inv)

    probes = Probes(inv.shared_dir, datasets, catalog=catalog)
    try:
        raw = llm.chat_with_tools(
            system=_SYSTEM,
            user=_context_message(inv, proposal),
            tools=all_tools(),
            dispatch=probes.dispatch,
            finalize_tool=FINALIZE_TOOL,
            max_steps=max_steps,
            on_step=on_step,
            cache_key=f"strategy:{inv.ipts}:{len(datasets)}:prop={int(bool(proposal and proposal.available))}",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Strategy LLM loop failed (%s); using deterministic fallback", e)
        return deterministic_strategy(datasets, inv)

    return _parse_strategy(raw, datasets, inv)


def _parse_strategy(raw: dict, datasets: list[Dataset], inv: FolderInventory) -> AnalysisStrategy:
    cs = (raw.get("curve_source") or "auto").lower()
    if cs not in ("combined", "individual", "auto"):
        cs = "auto"
    strat = AnalysisStrategy(
        experiment_summary=raw.get("experiment_summary", ""),
        science_goals=list(raw.get("science_goals") or []),
        curve_source=cs,
        curve_source_rationale=raw.get("curve_source_rationale", ""),
        report_outline=list(raw.get("report_outline") or []),
        caveats=list(raw.get("caveats") or []),
        open_questions=list(raw.get("open_questions") or []),
    )
    vd = raw.get("variant_decision") or {}
    strat.variant_decision = VariantDecision(
        variants_used=list(vd.get("variants_used") or _default_variants(inv)),
        compare=bool(vd.get("compare", False)),
        rationale=vd.get("rationale", ""),
    )
    valid_names = {d.output_name for d in datasets}
    for g in raw.get("groups") or []:
        members = [m for m in (g.get("members") or []) if m in valid_names]
        if not members:
            continue
        strat.groups.append(StrategyGroup(
            group_id=g.get("group_id") or g.get("label", "group"),
            label=g.get("label", "Group"),
            kind=g.get("kind", "single"),
            members=members,
            comparison=g.get("comparison", "iq1d"),
            ordering_key=g.get("ordering_key"),
            description=g.get("description", ""),
        ))
    for fp in raw.get("fit_plans") or []:
        model = fp.get("model")
        strat.fit_plans.append(FitPlan(
            group_id=fp.get("group_id", ""),
            should_fit=bool(fp.get("should_fit", False)),
            model=None if model in (None, "none") else model,
            q_min=fp.get("q_min"), q_max=fp.get("q_max"),
            rationale=fp.get("rationale", ""),
        ))
    if not strat.groups:  # safety net
        return deterministic_strategy(datasets, inv)
    return strat


def _default_variants(inv: FolderInventory) -> list[str]:
    return [p.name for p in inv.output_dirs] or ["output"]


# --------------------------------------------------------------------------- #
# Deterministic fallback (no LLM)
# --------------------------------------------------------------------------- #
def deterministic_strategy(datasets: list[Dataset], inv: FolderInventory) -> AnalysisStrategy:
    """Group by base sample; detect temperature series; one variant only."""
    variants = _default_variants(inv)
    primary = variants[-1] if len(variants) > 1 else variants[0]  # prefer last (often newest)

    science = [d for d in datasets if not d.is_standard and d.variant == primary]
    by_base: dict[str, list[Dataset]] = {}
    for d in science:
        by_base.setdefault(d.base, []).append(d)

    groups: list[StrategyGroup] = []
    fit_plans: list[FitPlan] = []
    for base, members in sorted(by_base.items()):
        temps = {m.temperature for m in members if m.temperature}
        kind = "temperature_series" if len(temps) >= 2 else ("config_set" if len(members) > 1 else "single")
        gid = f"grp_{base}"
        groups.append(StrategyGroup(
            group_id=gid,
            label=f"Sample {base}" + (" (temperature series)" if kind == "temperature_series" else ""),
            kind=kind,
            members=[m.output_name for m in members],
            comparison="iq1d",
            ordering_key="temperature" if kind == "temperature_series" else None,
            description="",
        ))
        fit_plans.append(FitPlan(group_id=gid, should_fit=False, model=None,
                                 rationale="deterministic mode: no fitting"))

    summary = (f"IPTS-{inv.ipts}: {len(science)} reduced datasets across "
               f"{len(by_base)} samples (deterministic grouping; no LLM reasoning).")
    return AnalysisStrategy(
        experiment_summary=summary,
        variant_decision=VariantDecision(variants_used=[primary], compare=False,
                                         rationale="deterministic: single newest variant"),
        groups=groups,
        fit_plans=fit_plans,
        report_outline=["Overview", "Sample Groups", "Methods & Caveats"],
        caveats=["Generated without LLM reasoning (--no-llm)."]
        + (["Multiple output variants present; used only " + primary] if len(variants) > 1 else []),
    )
