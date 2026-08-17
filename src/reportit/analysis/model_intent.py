"""Does the proposal actually ask for a model-based analysis?

Model fitting is only worth doing when the science question is a model parameter.
Running a sasmodels search on every experiment produces confident numbers that
answer nobody's question — and worse, a model can win on curve-shape agreement
while reproducing none of the structure that matters (a single broad component
smeared over several finer peaks, or a lamellar spacing matching no observed
peak). A wrong number in a report is more damaging than no number.

So model fitting is **opt-in**: it runs when the operator asks for it
(``--sasfit``), or when the proposal itself names the analysis it wants. If the
proposal says "we will determine Rg from the Guinier region", a Guinier analysis
is exactly what the report should contain, and this module finds that.

What a default run always does regardless: the curves, the qualitative
observations, and any analysis the proposal's goal implies — notably the peak /
repeat-distance measurement (see :mod:`.dspacing`), which is empirical and needs
no model choice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Proposal wording -> the sasmodels family it is asking for. Phrases are specific
# on purpose: under --no-llm the proposal summary can be raw PDF text, and a
# loose match would switch fitting on for every experiment.
_MODEL_TERMS: dict[str, tuple[str, ...]] = {
    "guinier": (r"guinier", r"radius of gyration", r"\bR_?g\b"),
    "correlation_length": (r"correlation length", r"\bmesh size\b",
                           r"ornstein[- ]zernike", r"blob size"),
    "porod": (r"porod", r"surface fractal", r"mass fractal",
              r"specific surface area", r"interfacial area"),
    "sphere": (r"\bsphere\b", r"spherical (?:particle|micelle|core)",
               r"core[- ]shell"),
    "cylinder": (r"\bcylinder\b", r"cylindrical (?:particle|micelle)", r"\brod-?like\b"),
    "polymer_excl_volume": (r"excluded volume", r"flory exponent", r"swollen chain"),
    "mono_gauss_coil": (r"gaussian coil", r"debye (?:function|model)", r"ideal chain"),
    "lamellar_stack_paracrystal": (r"lamellar", r"bilayer stack", r"multilamellar"),
}

# Wording that asks for model fitting in general, without naming a model.
_GENERIC = (r"model[- ]based (?:fit|analysis)", r"fit(?:ted|ting)? (?:to|with) a model",
            r"form factor", r"structure factor", r"sasview", r"sasmodels",
            r"quantitative model")


@dataclass
class ModelIntent:
    """Whether the proposal calls for model fitting, and which models."""

    wanted: bool = False
    models: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    generic: bool = False

    def reason(self) -> str:
        if not self.wanted:
            return ""
        if self.models:
            return ("the proposal names the analysis it wants ("
                    + ", ".join(self.terms[:6]) + ")")
        return "the proposal asks for a model-based analysis"


def _proposal_text(proposal) -> str:
    if proposal is None or not getattr(proposal, "available", False):
        return ""
    chunks = []
    for attr in ("title", "abstract_summary"):
        v = getattr(proposal, attr, None)
        if v:
            chunks.append(str(v))
    chunks += [str(g) for g in (getattr(proposal, "science_goals", None) or [])]
    for h in (getattr(proposal, "hypotheses", None) or []):
        chunks.append(str(getattr(h, "text", "")))
        chunks.append(str(getattr(h, "expected_signature", "")))
    return "\n".join(c for c in chunks if c)


def detect(proposal) -> ModelIntent:
    """Decide from the proposal whether a model-based analysis is called for."""
    blob = _proposal_text(proposal)
    if not blob:
        return ModelIntent()
    models, terms = [], []
    for model, patterns in _MODEL_TERMS.items():
        for pat in patterns:
            m = re.search(pat, blob, re.IGNORECASE)
            if m:
                if model not in models:
                    models.append(model)
                terms.append(m.group(0).lower())
                break
    generic = any(re.search(p, blob, re.IGNORECASE) for p in _GENERIC)
    wanted = bool(models or generic)
    return ModelIntent(wanted=wanted, models=models,
                       terms=sorted(set(terms)), generic=generic)


def prompt_hint(intent: ModelIntent) -> str:
    """Tell the model selector which analysis the proposal actually asked for."""
    if not intent.wanted or not intent.models:
        return ""
    return (
        "\nTHE PROPOSAL NAMES THE ANALYSIS IT WANTS: "
        + ", ".join(intent.models)
        + f" (from the wording: {', '.join(intent.terms[:6])}). Put those models "
        "FIRST among your candidates and report their parameters, because that is "
        "the number the experiment was proposed to measure. You may still add a "
        "better-fitting alternative, but say plainly if the proposal's model does "
        "not describe the data — that is itself the result."
    )
