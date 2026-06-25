"""Build a catalog of SasView/sasmodels models for LLM-driven model selection.

Sourced directly from the installed sasmodels package (always in sync), so we
don't need to vendor SasView's web docs. A curated short-list of the models most
relevant to SANS solution/soft-matter work is surfaced first.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Models most relevant to polymer/colloid/solution SANS — shown to the selector
# first so it doesn't have to reason over all ~80. It may still pick others.
PRIORITY_MODELS = [
    "mono_gauss_coil", "poly_gauss_coil", "gauss_lorentz_gel", "dab",
    "correlation_length", "broad_peak", "guinier", "guinier_porod",
    "power_law", "porod", "sphere", "polydisperse_sphere", "fractal",
    "mass_fractal", "surface_fractal", "fuzzy_sphere", "core_shell_sphere",
    "ellipsoid", "cylinder", "flexible_cylinder", "star_polymer",
    "polymer_excl_volume", "two_power_law", "unified_power_Rg", "lorentz",
    "teubner_strey", "peak_lorentz",
]


def _model_entry(name: str) -> dict | None:
    try:
        from sasmodels.core import load_model_info
        info = load_model_info(name)
    except Exception as e:  # noqa: BLE001
        logger.debug("skip model %s: %s", name, e)
        return None
    params = []
    for p in info.parameters.kernel_parameters:
        params.append({
            "name": p.name,
            "default": getattr(p, "default", None),
            "limits": list(getattr(p, "limits", (None, None))),
            "units": getattr(p, "units", ""),
            "desc": (getattr(p, "description", "") or "")[:80],
        })
    desc = (getattr(info, "description", "") or "").strip().replace("\n", " ")
    return {"name": name, "title": getattr(info, "name", name),
            "description": desc[:400], "parameters": params}


def list_models() -> list[str]:
    try:
        from sasmodels.core import list_models as _lm
        return list(_lm())
    except Exception as e:  # noqa: BLE001
        logger.warning("could not list sasmodels: %s", e)
        return []


def short_catalog() -> list[dict]:
    """Name + one-line description + parameter names, priority models first."""
    all_models = list_models()
    ordered = [m for m in PRIORITY_MODELS if m in all_models]
    ordered += [m for m in all_models if m not in PRIORITY_MODELS]
    out = []
    for name in ordered:
        e = _model_entry(name)
        if not e:
            continue
        out.append({
            "name": e["name"],
            "description": _first_sentence(e["description"]),
            "parameters": [p["name"] for p in e["parameters"]],
        })
    return out


def model_detail(name: str) -> dict | None:
    """Full parameter detail (defaults, limits, units) for one model."""
    return _model_entry(name)


def _first_sentence(s: str) -> str:
    s = s.strip()
    for sep in (". ", ".\n"):
        if sep in s:
            return s.split(sep)[0][:160]
    return s[:160]
