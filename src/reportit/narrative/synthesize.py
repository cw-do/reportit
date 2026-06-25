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
    "You are a SANS expert writing the results subsection for one group of EQSANS "
    "measurements. Given the group's metadata, per-dataset metrics, any fit, and a "
    "visual description of the actual I(Q) overlay plot, write 2-4 sentences of "
    "factual scientific observation: the shape/Q-dependence seen in the plot, how "
    "the curves differ across the series (temperature/concentration), any peaks, "
    "plateaus, low-Q upturns or power-law regions, and what it means in the "
    "experiment's context. Ground statements in what the plot shows. No markdown, "
    "no headings — just prose."
)

_OBS_VISION_SYS = (
    "You are a SANS expert visually inspecting a log-log I(Q) overlay plot for a "
    "group of related samples. Describe concretely what you SEE: overall shape and "
    "Q-dependence, approximate power-law slopes, any low-Q plateau or upturn, peaks "
    "or knees, and — importantly — how the curves differ from each other across the "
    "series (do they shift up/down, change slope, move a feature?). Be specific."
)


def _vision_question(gr, context: str) -> str:
    return (
        f"Experiment context: {context[:800]}\n"
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
        "metrics": [
            {"name": a.output_name, "variant": a.variant, "n": a.n_points,
             "q_min": a.q_min, "q_max": a.q_max,
             "low_q_slope": a.low_q_slope, "high_q_slope": a.high_q_slope,
             "fit": ({"kind": a.fit.kind, "params": a.fit.params,
                      "r2": a.fit.r_squared} if a.fit and a.fit.ok else None),
             "flags": a.flags}
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
    payload["experiment_context"] = context[:800]
    payload["plot_observation"] = vision_note
    try:
        return llm.chat(_GROUP_SYS, json.dumps(payload, default=str),
                        max_tokens=2500,
                        cache_key=f"obs:{gr.group.group_id}:{len(gr.analyses)}:v2")
    except Exception as e:  # noqa: BLE001
        logger.warning("group observation failed: %s", e)
        return vision_note or _deterministic_group_text(gr)


def _deterministic_group_text(gr: GroupReport) -> str:
    parts = [f"{gr.group.label}: {len(gr.analyses)} dataset(s)."]
    for a in gr.analyses[:6]:
        seg = f" {a.output_name} spans Q in [{a.q_min:.3g}, {a.q_max:.3g}] 1/A"
        if a.low_q_slope is not None:
            seg += f", low-Q log-log slope ~ {a.low_q_slope:.2f}"
        parts.append(seg + ".")
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
