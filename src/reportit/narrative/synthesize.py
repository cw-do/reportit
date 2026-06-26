"""Generate prose for the report via the LLM.

Per-group observations, a global overview+discussion, and hypothesis checks.
Degrades to deterministic templated text when no LLM is available.
"""

from __future__ import annotations

import json
import logging

from ..llm import LLMClient
from ..models import (
    AnalysisStrategy,
    GroupReport,
    HypothesisCheck,
    ProposalInfo,
)

logger = logging.getLogger(__name__)

_GROUP_SYS = (
    "You are a SANS expert writing a QUALITATIVE results subsection for one group of "
    "EQSANS measurements (no model fitting in this section — that comes later). "
    "Given the group's metadata and a visual description of the actual I(Q) "
    "overlay, write 3-6 sentences that (a) describe what the plot shows and how the "
    "curves differ across the series (temperature/concentration), and (b) offer "
    "careful physical interpretation and hypotheses. "
    "IMPORTANT: do NOT quote specific power-law slope numbers — a slope depends "
    "strongly on the Q-range chosen and that quantitative work belongs to the later "
    "fitting section. Describe shape QUALITATIVELY instead (steeper/shallower, a "
    "plateau, a knee, a low-Q upturn, a peak, curves shifting up/down or crossing "
    "over). Be nuanced about meaning: a low-Q upturn may be aggregation, a network/"
    "large-scale correlation, or genuine large-object scattering — present options "
    "rather than asserting one. Note the high-Q flat region is the incoherent "
    "background and the 1-2 lowest-Q points may be beam-stop/mask artifacts. Tie "
    "observations to the experiment's goals. Ground every statement in the plot. "
    "No markdown, no headings — just prose."
)

_OBS_VISION_SYS = (
    "You are a SANS expert visually inspecting a log-log I(Q) overlay plot for a "
    "group of related samples. Describe concretely what you SEE QUALITATIVELY: the "
    "overall shape, any low-Q plateau/upturn, peaks or knees, the high-Q flat "
    "(incoherent background) level, and — importantly — how the curves differ "
    "across the series (shift up/down, get steeper/shallower, move a feature, cross "
    "over?). Flag the 1-2 lowest-Q points if they look like masking artifacts. Do "
    "NOT quote specific numerical power-law slopes — they depend on the Q-range and "
    "are handled later by fitting; describe steepness in relative, qualitative terms."
)


def _vision_question(gr, context: str) -> str:
    return (
        f"Experiment context: {context[:6000]}\n"
        f"This plot is: {gr.group.label} (a {gr.group.kind}). "
        "Describe what the overlaid I(Q) curves show and how they differ across the series."
    )

_GLOBAL_SYS = (
    "You are a SANS expert writing the Overview and Discussion of an EQSANS "
    "experiment report. Be factual and concise. Return JSON with keys: "
    '"overview" (one paragraph introducing the experiment and what was measured), '
    '"discussion" (one or two paragraphs synthesizing the findings across groups), '
    'and "hypothesis_checks" (a list of {"hypothesis","verdict","evidence",'
    '"confidence"} where verdict is one of supported|not_supported|inconclusive|'
    'no_data and confidence is high|medium|low). If there are no proposal '
    "hypotheses, return an empty hypothesis_checks list."
)


def _group_payload(gr: GroupReport) -> dict:
    return {
        "label": gr.group.label,
        "kind": gr.group.kind,
        "description": gr.group.description,
        "ordering_key": gr.group.ordering_key,
        # Q-range/point count only — NO precomputed slopes (they depend on the
        # unstated Q-range and would invite misleading slope claims here).
        "datasets": [
            {"name": a.output_name, "variant": a.variant, "n_points": a.n_points,
             "q_min": a.q_min, "q_max": a.q_max, "flags": a.flags}
            for a in gr.analyses
        ],
    }


def observe_group(gr: GroupReport, llm: LLMClient | None, context: str = "") -> str:
    if llm is None:
        return _deterministic_group_text(gr)

    # 1) visually inspect the actual overlay plot (multimodal)
    vision_note = ""
    fig = next((f for f in gr.figures if f.label.endswith("_iq")), None)
    if fig is None and gr.figures:
        fig = gr.figures[0]
    if fig is not None:
        try:
            vision_note = llm.chat_vision(
                _OBS_VISION_SYS, _vision_question(gr, context), fig.path,
                max_tokens=2000, cache_key=f"obsvis:{gr.group.group_id}")
        except Exception as e:  # noqa: BLE001
            logger.warning("group vision obs failed: %s", e)

    # 2) write the observation from metrics + the visual description
    payload = _group_payload(gr)
    payload["experiment_context"] = context[:6000]
    payload["plot_observation"] = vision_note
    try:
        return llm.chat(_GROUP_SYS, json.dumps(payload, default=str),
                        max_tokens=6000,
                        cache_key=f"obs:{gr.group.group_id}:{len(gr.analyses)}:v2")
    except Exception as e:  # noqa: BLE001
        logger.warning("group observation failed: %s", e)
        return vision_note or _deterministic_group_text(gr)


def _deterministic_group_text(gr: GroupReport) -> str:
    # Qualitative/descriptive only — no slopes (they depend on the Q-range used).
    parts = [f"{gr.group.label}: {len(gr.analyses)} dataset(s)."]
    qs = [a.q_min for a in gr.analyses if a.q_min] + [a.q_max for a in gr.analyses if a.q_max]
    if qs:
        parts.append(f" Data span Q in [{min(qs):.3g}, {max(qs):.3g}] 1/A.")
    parts.append(" See the overlaid I(Q) curves for the qualitative comparison "
                 "across the series.")
    return "".join(parts)


def global_narrative(
    strategy: AnalysisStrategy,
    group_reports: list[GroupReport],
    proposal: ProposalInfo,
    llm: LLMClient | None,
) -> tuple[str, str, list[HypothesisCheck]]:
    if llm is None:
        overview = strategy.experiment_summary or "EQSANS experiment summary."
        return overview, "", []

    payload = {
        "experiment_summary": strategy.experiment_summary,
        "science_goals": strategy.science_goals,
        "variant_decision": {
            "variants_used": strategy.variant_decision.variants_used,
            "compare": strategy.variant_decision.compare,
            "rationale": strategy.variant_decision.rationale,
        },
        "proposal_hypotheses": [
            {"text": h.text, "expected_signature": h.expected_signature}
            for h in (proposal.hypotheses if proposal else [])
        ],
        "groups": [
            {"label": gr.group.label, "kind": gr.group.kind,
             "observations_seed": _group_payload(gr)}
            for gr in group_reports
        ],
    }
    try:
        data = llm.chat_json(_GLOBAL_SYS, json.dumps(payload, default=str),
                             max_tokens=8000,
                             cache_key=f"global:{strategy.experiment_summary[:40]}:{len(group_reports)}")
    except Exception as e:  # noqa: BLE001
        logger.warning("global narrative failed: %s", e)
        return strategy.experiment_summary, "", []

    checks = []
    for c in data.get("hypothesis_checks") or []:
        if isinstance(c, dict):
            checks.append(HypothesisCheck(
                hypothesis=c.get("hypothesis", ""),
                verdict=c.get("verdict", "inconclusive"),
                evidence=c.get("evidence", ""),
                confidence=c.get("confidence", "low"),
            ))
    return data.get("overview", strategy.experiment_summary), data.get("discussion", ""), checks
