"""Turn raw proposal text into a structured ProposalInfo via the LLM."""

from __future__ import annotations

import logging
from pathlib import Path

from ..llm import LLMClient
from ..models import Hypothesis, ProposalInfo
from . import extract

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a neutron-scattering (SANS) expert reading a beamtime proposal / "
    "statement of research for an EQSANS experiment. Extract the scientific "
    "intent so a downstream tool can check whether the measured data supports it. "
    "Be concrete and faithful to the text; do not invent samples not mentioned."
)


def _prompt(text: str, sample_bases: list[str]) -> str:
    return (
        "Proposal text follows. Summarize it as JSON with keys:\n"
        '  "title" (string|null), "pi" (string|null),\n'
        '  "abstract_summary" (2-4 sentence plain summary),\n'
        '  "science_goals" (list of short strings),\n'
        '  "hypotheses" (list of objects {"text", "expected_signature"} where '
        "expected_signature describes what to look for in SANS I(Q) data, e.g. "
        "'Guinier radius Rg increases with temperature' or 'power-law slope ~ -4 "
        "indicating sharp interfaces'),\n"
        '  "sample_descriptions" (object mapping any sample label/code you can '
        "infer to a short description).\n\n"
        f"The reduced-data sample base names found on disk are: {sample_bases}. "
        "Map descriptions to these where you can; abbreviations are common.\n\n"
        "=== PROPOSAL TEXT ===\n" + text[:48000]
    )


def summarize(pdf_paths: list[Path], llm: LLMClient | None) -> ProposalInfo:
    text = extract.extract_many(pdf_paths)
    info = ProposalInfo(available=bool(text.strip()), raw_text_chars=len(text))
    if not text.strip():
        logger.info("No extractable proposal text.")
        return info
    if llm is None:
        info.abstract_summary = text[:1500]
        return info

    try:
        data = llm.chat_json(_SYSTEM, _prompt(text, []),
                             cache_key=f"proposal:{','.join(p.name for p in pdf_paths)}")
    except Exception as e:  # noqa: BLE001
        logger.warning("Proposal summarization failed: %s", e)
        info.abstract_summary = text[:1500]
        return info

    info.title = data.get("title")
    info.pi = data.get("pi")
    info.abstract_summary = data.get("abstract_summary", "")
    info.science_goals = list(data.get("science_goals") or [])
    info.sample_descriptions = dict(data.get("sample_descriptions") or {})
    for h in data.get("hypotheses") or []:
        if isinstance(h, dict):
            info.hypotheses.append(Hypothesis(
                text=h.get("text", ""),
                expected_signature=h.get("expected_signature", ""),
            ))
        elif isinstance(h, str):
            info.hypotheses.append(Hypothesis(text=h))
    return info
