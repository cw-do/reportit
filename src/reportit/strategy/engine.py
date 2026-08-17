"""The agentic strategy loop: inventory + proposal → LLM probes → AnalysisStrategy.

Also provides a deterministic fallback strategy for --no-llm runs.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
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
{variant_rule}
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

# The variant decision only exists when reportit discovered the data folders
# itself. With --data the user has already chosen, so the rule is replaced by an
# instruction NOT to spend tool calls re-litigating it.
_VARIANT_RULE_DISCOVERED = """\
  - If there are multiple output directories (e.g. output vs output_mask4 with \
different detector masks), decide which to use, or whether to compare them, and \
WHY (read NOTE.md and compare). STRONGLY prefer a variant that HAS combined/merged \
extended-Q profiles (see the per-output-dir coverage in the inventory) — a variant \
with only per-config 1D files and no merged data gives much worse plots and fits, \
so do not pick it when a merged-bearing variant exists. Do not compare a \
merged-bearing variant against one that lacks merged data."""

_VARIANT_RULE_FIXED = """\
  - The reduced-data directory is ALREADY FIXED (the user passed --data) and is \
listed in the inventory. Do NOT look for, list, or evaluate any other output \
directory, and do not spend tool calls on the variant question — it is settled. \
Every dataset you can use is already returned by list_datasets. Leave \
variant_decision empty and spend your effort on the science: what the samples \
are, how to group them, and which fits make sense."""


def _system_prompt(inv: FolderInventory) -> str:
    return _SYSTEM.replace(
        "{variant_rule}",
        _VARIANT_RULE_FIXED if inv.data_dirs_fixed else _VARIANT_RULE_DISCOVERED)


def _context_message(inv: FolderInventory, proposal: ProposalInfo) -> str:
    lines = []
    # Reference knowledge reaches the PLANNING agent too, not just the fitter:
    # how to group samples and what is worth analysing is exactly the kind of
    # judgement the operator teaches through the knowledge notes.
    from ..analysis import knowledge
    kb = knowledge.load_knowledge(
        stage="strategy",
        context=" ".join([inv.as_text()[:4000],
                          getattr(proposal, "abstract_summary", "") or "",
                          " ".join(getattr(proposal, "science_goals", None) or [])]))
    if kb:
        lines += ["=== REFERENCE KNOWLEDGE (general; applies to every experiment) ===",
                  kb, ""]
    lines += ["=== FOLDER INVENTORY ===", inv.as_text(), "", "=== PROPOSAL SUMMARY ==="]
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
    guide=None,
) -> AnalysisStrategy:
    if llm is None:
        return deterministic_strategy(datasets, inv)

    from .. import guidance

    # the reduced-data dirs are readable too — with --data they may lie outside
    # the target tree, and the agent still needs to inspect them.
    probes = Probes(inv.shared_dir, datasets, catalog=catalog,
                    extra_roots=list(inv.output_dirs))
    # the data dirs are part of the cache identity: pointing --data somewhere else
    # must re-derive the strategy, not reuse the previous folder's answer.
    dirs_key = hashlib.sha1(
        "|".join(sorted(str(p) for p in inv.output_dirs)).encode()).hexdigest()[:10]
    # knowledge is part of the cache identity: teaching a new lesson must change
    # the strategy, not replay the pre-lesson answer
    from ..analysis import knowledge
    kb_key = knowledge.digest("strategy")
    try:
        raw = llm.chat_with_tools(
            system=_system_prompt(inv),
            user=_context_message(inv, proposal) + guidance.hint_block(guide, "strategy"),
            tools=all_tools(),
            dispatch=probes.dispatch,
            finalize_tool=FINALIZE_TOOL,
            max_steps=max_steps,
            on_step=on_step,
            cache_key=f"strategy:{inv.ipts}:{len(datasets)}:dirs={dirs_key}:"
                      f"fixed={int(inv.data_dirs_fixed)}:"
                      f"guide={guidance.digest(guide)}:kb={kb_key}:"
                      f"prop={int(bool(proposal and proposal.available))}",
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
    return _apply_grouping_guard(strat, datasets, inv)


def _default_variants(inv: FolderInventory) -> list[str]:
    from ..discovery.inventory import variant_labels
    return sorted(variant_labels(inv.output_dirs).values()) or ["output"]


# --------------------------------------------------------------------------- #
# Grouping guard: the agentic strategy loop sometimes under-groups (finalises
# after inspecting only a couple of series, leaving most samples ungrouped).
# When coverage is poor, replace its groups with a deterministic series grouping
# so EVERY science sample appears and related samples are compared. Merged
# profiles are always preferred downstream, so this also restores merged plots.
# --------------------------------------------------------------------------- #
_SERIES_HYPHEN = re.compile(r"^(.*)-(\d+(?:\.\d+)?)$")   # salinity: 260K.D2O.5-0
_SERIES_TRAIL = re.compile(r"^(.*?\D)(\d+(?:\.\d+)?)$")  # concentration: dP2VPNO.D2O.25, pb20


def _series_stem(base: str) -> tuple[str, str | None]:
    """Split a sample base into (series_stem, series_value) by stripping the one
    varying series token — a salinity '-N' or a trailing concentration 'N'.
    Config/thickness tokens after '_' (e.g. '1mm') are kept in the stem so
    different thicknesses do not collapse together."""
    if not base:
        return base, None
    parts = base.split("_")
    head = parts[0]
    val = None
    m = _SERIES_HYPHEN.match(head) or _SERIES_TRAIL.match(head)
    if m:
        head, val = m.group(1), m.group(2)
    return "_".join([head] + parts[1:]), val


def _robust_groups(science: list[Dataset]) -> list[StrategyGroup]:
    """Deterministic grouping: bucket by (series_stem, temperature); one member
    per distinct sample base (preferring the merged-bearing one). Covers ALL
    science datasets."""
    buckets: dict[tuple, dict] = defaultdict(dict)  # (stem, temp) -> {base: ds}
    for d in science:
        if _is_background(d.base):
            continue
        stem, _ = _series_stem(d.base or d.output_name)
        by_base = buckets[(stem, d.temperature or "")]
        cur = by_base.get(d.base)
        if cur is None or (d.merged_path and not cur.merged_path):
            by_base[d.base] = d
    groups: list[StrategyGroup] = []
    for (stem, temp), by_base in buckets.items():
        reps = list(by_base.values())
        n = len(reps)
        temps = {m.temperature for m in reps if m.temperature}
        kind = ("concentration_series" if n >= 2
                else "temperature_series" if len(temps) >= 2 else "single")
        pretty = stem.rstrip("._-")
        label = pretty + (" series" if kind == "concentration_series" else "") \
            + (f" ({temp})" if temp else "")
        groups.append(StrategyGroup(
            group_id=_safe_id(f"{stem}_{temp}"), label=label, kind=kind,
            members=[m.output_name for m in reps], comparison="iq1d",
            ordering_key="concentration" if kind == "concentration_series" else None,
            description=""))
    groups.sort(key=lambda g: g.label)
    return groups


def _is_background(base: str) -> bool:
    """Obvious solvent / empty-cell / background samples — not science groups."""
    b = (base or "").lower()
    return (b in {"d2o", "h2o", "banjo", "empty", "emptycell", "blank"}
            or b.startswith(("bkg", "bkgd", "background", "buffer")))


def _apply_grouping_guard(strat: AnalysisStrategy, datasets: list[Dataset],
                          inv: FolderInventory) -> AnalysisStrategy:
    used = set(strat.variant_decision.variants_used or _default_variants(inv))
    science = [d for d in datasets if not d.is_standard and d.variant in used]
    bases = {d.base for d in science}
    if len(bases) < 3:
        return strat  # too few samples for grouping to matter
    name_to_base = {d.output_name: d.base for d in science}
    covered = {name_to_base[m] for g in strat.groups for m in g.members
               if m in name_to_base}
    frac = len(covered) / len(bases)
    if frac >= 0.8:
        return strat  # LLM grouping covers the samples well enough
    robust = _robust_groups(science)
    if robust:
        logger.warning("grouping guard: LLM covered %.0f%% of samples (%d/%d); "
                       "using deterministic series grouping (%d groups)",
                       100 * frac, len(covered), len(bases), len(robust))
        strat.groups = robust
        strat.caveats.append(
            f"Grouping guard: the strategy step covered only {frac:.0%} of the "
            f"measured samples, so a deterministic series grouping ({len(robust)} "
            "groups) was used instead to compare all related samples.")
    return strat


def _safe_id(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_").lower() or "group"


# --------------------------------------------------------------------------- #
# Deterministic fallback (no LLM)
# --------------------------------------------------------------------------- #
def deterministic_strategy(datasets: list[Dataset], inv: FolderInventory) -> AnalysisStrategy:
    """Group by base sample; detect temperature series; one variant only."""
    variants = _default_variants(inv)
    primary = variants[-1] if len(variants) > 1 else variants[0]  # prefer last (often newest)

    science = [d for d in datasets if not d.is_standard and d.variant == primary]
    groups = _robust_groups(science)   # series grouping (salinity/concentration/temperature)
    fit_plans = [FitPlan(group_id=g.group_id, should_fit=False, model=None,
                         rationale="deterministic mode: no fitting") for g in groups]
    n_bases = len({d.base for d in science})
    summary = (f"IPTS-{inv.ipts}: {len(science)} reduced datasets across "
               f"{n_bases} samples (deterministic series grouping; no LLM reasoning).")
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
