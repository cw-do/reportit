"""Interpret a user's free-text analysis directive and route it to the pipeline
stages it affects.

`reportit ... --userguide "use merged files only for the analysis and summary"`

Rather than pasting the sentence into every prompt and hoping each stage honours
it, the text is interpreted ONCE into a structured :class:`Guidance`:

  * the parts that are really a **data-selection rule** ("merged files only",
    "skip the porsil standard", "only the 30C runs") become concrete filters
    applied in code — the next stage receives an already-narrowed dataset list,
    not a request it might ignore;
  * the parts that are genuinely advisory ("group by salinity", "prefer
    correlation length", "keep the summary short") are routed as hints to the
    strategy / fitting / narrative prompts, and only to those.

Everything the guidance did is recorded on the Guidance object so the report can
state plainly how the user's instruction changed the analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

from .models import Dataset, Guidance

logger = logging.getLogger(__name__)

# Stages the guidance can be routed to (also the vocabulary given to the LLM).
STAGES = ("data_selection", "strategy", "fitting", "narrative")

_SYS = (
    "You route a neutron-scattering analyst's free-text instruction to the right "
    "stage of an automated SANS report pipeline. The pipeline stages are:\n"
    "  data_selection — WHICH reduced datasets/files are analyzed at all;\n"
    "  strategy       — how datasets are grouped and compared, which are science "
    "vs standards, whether to fit;\n"
    "  fitting        — which sasmodels are tried and how fits are judged;\n"
    "  narrative      — how the written report/summary is worded.\n\n"
    "Decide which stages the instruction affects, and express any DATA-SELECTION "
    "part as concrete, mechanical rules so it can be enforced in code:\n"
    "  merged_only      — true if the user wants only merged/stitched/combined "
    "extended-Q profiles (a dataset qualifies when it HAS a merged file).\n"
    "  include_patterns — case-insensitive regular expressions; a dataset is kept "
    "when its name/base matches ANY of them. Use ONLY for an explicit positive "
    "restriction ('only the 30C runs', 'just the dP2VPNO samples'). Leave empty "
    "otherwise — an empty list means 'keep everything'.\n"
    "  exclude_patterns — case-insensitive regular expressions; datasets matching "
    "ANY are dropped ('skip porsil', 'ignore the banjo background').\n"
    "Write patterns against sample names like 'dP2VPNO.D2O.25_1mm' or '260K.D2O.5-0' "
    "— you are shown real examples. Be conservative: a pattern that is too broad "
    "silently deletes data. If the instruction says nothing about selection, leave "
    "all three empty/false.\n\n"
    "For the advisory stages write a SHORT imperative hint (one or two sentences, "
    "empty string if that stage is unaffected) that will be appended to that "
    "stage's prompt. Do not restate the whole instruction in every hint.\n"
    "Also give a one-sentence plain-language 'interpretation' of what you took the "
    "instruction to mean — the user sees this, so make it faithful, and say so if "
    "the instruction is ambiguous or you could not act on part of it."
)


def _schema_hint() -> str:
    return (
        'Return JSON: {"interpretation": <one sentence>, '
        '"stages": [<subset of ' + ", ".join(STAGES) + '>], '
        '"data_selection": {"merged_only": <bool>, "include_patterns": [<regex>], '
        '"exclude_patterns": [<regex>]}, '
        '"strategy_hint": <text or "">, "fitting_hint": <text or "">, '
        '"narrative_hint": <text or "">}'
    )


def digest(g: Guidance | None) -> str:
    """Short stable hash of the guidance, for mixing into LLM cache keys so a
    changed instruction never replays a previous run's answers."""
    if g is None or not g.active:
        return "none"
    return hashlib.sha1(g.text.strip().encode()).hexdigest()[:10]


def interpret(text: str, llm, datasets: list[Dataset], cache_key: str | None = None) -> Guidance:
    """Interpret ``text`` into a routed Guidance. Falls back to a keyword reading
    when there is no LLM or the call fails — never raises."""
    g = Guidance(text=(text or "").strip())
    if not g.active:
        return g
    if llm is None:
        return _keyword_guidance(g)

    names = sorted({d.output_name for d in datasets})
    sample = names[:60]
    prompt = (
        f"INSTRUCTION FROM THE USER:\n{g.text}\n\n"
        f"DATASET NAMES IN THIS EXPERIMENT ({len(names)} total, showing "
        f"{len(sample)}):\n{json.dumps(sample)}\n\n"
        f"How many have a merged/combined extended-Q profile: "
        f"{sum(1 for d in datasets if d.merged_path)} of {len(datasets)}.\n\n"
        + _schema_hint()
    )
    try:
        raw = llm.chat_json(_SYS, prompt, max_tokens=2000,
                            cache_key=cache_key or f"guidance:{digest(g)}")
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not interpret --userguide with the LLM (%s); "
                       "falling back to keyword reading.", e)
        return _keyword_guidance(g)

    sel = raw.get("data_selection") or {}
    g.interpretation = str(raw.get("interpretation") or "").strip()
    g.stages = [s for s in (raw.get("stages") or []) if s in STAGES]
    g.merged_only = bool(sel.get("merged_only"))
    g.include_patterns = _clean_patterns(sel.get("include_patterns"))
    g.exclude_patterns = _clean_patterns(sel.get("exclude_patterns"))
    g.strategy_hint = str(raw.get("strategy_hint") or "").strip()
    g.fitting_hint = str(raw.get("fitting_hint") or "").strip()
    g.narrative_hint = str(raw.get("narrative_hint") or "").strip()
    if not g.stages:  # infer from what it actually produced
        g.stages = _infer_stages(g)
    return g


def _clean_patterns(items) -> list[str]:
    """Keep only patterns that actually compile — a bad regex must not crash a run."""
    out = []
    for p in items or []:
        p = str(p).strip()
        if not p:
            continue
        try:
            re.compile(p, re.IGNORECASE)
        except re.error as e:
            logger.warning("Ignoring invalid --userguide pattern %r: %s", p, e)
            continue
        out.append(p)
    return out


def _infer_stages(g: Guidance) -> list[str]:
    stages = []
    if g.merged_only or g.include_patterns or g.exclude_patterns:
        stages.append("data_selection")
    for key, name in ((g.strategy_hint, "strategy"), (g.fitting_hint, "fitting"),
                      (g.narrative_hint, "narrative")):
        if key:
            stages.append(name)
    return stages


def _keyword_guidance(g: Guidance) -> Guidance:
    """Deterministic reading for --no-llm (or when interpretation fails).

    Only the unambiguous, high-value case is handled mechanically: restricting
    the analysis to merged/combined profiles. Anything else is passed through as
    a hint so the instruction is at least visible to the LLM stages, and the
    limitation is stated in the interpretation.
    """
    low = g.text.lower()
    merged_words = ("merged", "combined", "stitched")
    only_words = ("only", "just", "exclusively", "nothing but")
    if any(w in low for w in merged_words) and any(w in low for w in only_words):
        g.merged_only = True
        g.interpretation = ("Read without an LLM: restricting the analysis to "
                            "datasets that have a merged/combined extended-Q profile.")
    else:
        g.interpretation = ("Read without an LLM: no mechanical data-selection rule "
                            "recognised; the instruction is passed to the analysis "
                            "steps as free text.")
    g.strategy_hint = g.fitting_hint = g.narrative_hint = g.text
    g.stages = _infer_stages(g)
    return g


# --------------------------------------------------------------------------- #
# applying the deterministic part
# --------------------------------------------------------------------------- #
def apply_to_datasets(g: Guidance, datasets: list[Dataset]) -> list[Dataset]:
    """Apply the data-selection rules, returning the datasets to analyze.

    Records what happened on ``g``. If the rules would discard EVERYTHING the
    filter is abandoned and all datasets are kept: an instruction that leaves
    nothing to analyze is far more likely to be a misreading than the user's
    intent, and an empty report hides the problem.
    """
    if not g.active or not (g.merged_only or g.include_patterns or g.exclude_patterns):
        return datasets

    inc = [re.compile(p, re.IGNORECASE) for p in g.include_patterns]
    exc = [re.compile(p, re.IGNORECASE) for p in g.exclude_patterns]

    kept, dropped = [], []
    for d in datasets:
        hay = " ".join(str(x) for x in (d.output_name, d.base, d.oncat_title or "") if x)
        if g.merged_only and not d.merged_path:
            dropped.append(d); continue
        if inc and not any(r.search(hay) for r in inc):
            dropped.append(d); continue
        if exc and any(r.search(hay) for r in exc):
            dropped.append(d); continue
        kept.append(d)

    if not kept:
        g.applied_notes.append(
            f"Data-selection rules from --userguide would have removed all "
            f"{len(datasets)} datasets, so they were NOT applied — the instruction "
            f"was probably read too narrowly. Analyzed everything instead.")
        logger.warning("%s", g.applied_notes[-1])
        return datasets

    if dropped:
        bits = []
        if g.merged_only:
            bits.append("only datasets with a merged/combined extended-Q profile")
        if g.include_patterns:
            bits.append("names matching " + ", ".join(repr(p) for p in g.include_patterns))
        if g.exclude_patterns:
            bits.append("excluding names matching "
                        + ", ".join(repr(p) for p in g.exclude_patterns))
        g.applied_notes.append(
            f"--userguide data selection kept {len(kept)} of {len(datasets)} "
            f"datasets ({'; '.join(bits)}).")
        logger.info("%s", g.applied_notes[-1])

    g.kept_names = [d.output_name for d in kept]
    g.dropped_names = [d.output_name for d in dropped]
    return kept


# --------------------------------------------------------------------------- #
# routing the advisory part into prompts
# --------------------------------------------------------------------------- #
def hint_block(g: Guidance | None, stage: str) -> str:
    """The text to append to ``stage``'s prompt — empty when unaffected."""
    if g is None or not g.active:
        return ""
    hint = {"strategy": g.strategy_hint, "fitting": g.fitting_hint,
            "narrative": g.narrative_hint}.get(stage, "")
    if not hint:
        return ""
    return ("\n\nUSER INSTRUCTION (from --userguide; follow it unless it "
            f"contradicts the measured data):\n{hint}\n"
            f"The user's original words were: \"{g.text}\"")


def summary_lines(g: Guidance | None) -> list[str]:
    """Lines describing the guidance and its effect, for the report's caveats."""
    if g is None or not g.active:
        return []
    out = [f'User guidance (--userguide): "{g.text}"']
    if g.interpretation:
        out.append(f"Interpreted as: {g.interpretation}")
    if g.stages:
        out.append("Applied to: " + ", ".join(g.stages).replace("_", " "))
    out.extend(g.applied_notes)
    return out
