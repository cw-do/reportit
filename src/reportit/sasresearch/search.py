"""Robust multi-start fitting.

The old fitter ran ONE local optimisation from ONE data-driven seed and only
escalated to a global search when it looked poor.  For models like
correlation_length / lorentz / teubner_strey that is fragile: a single bad
starting point lands in a local minimum and stays there.

Here we run each model from MANY diverse starting points — the size parameter
swept around the data's knee, the power-law exponents swept over their physical
range — and keep whichever converged best by the shape-aware score, not by
chi^2 alone.  Each individual fit still goes through the existing
``sasfit.fit_curve`` (which itself does a local fit with a global-search
fallback), so this is a wrapper, not a reimplementation.
"""

from __future__ import annotations

import copy
import logging
import warnings

import numpy as np

from ..analysis import sasfit, sascatalog
from . import shape

logger = logging.getLogger(__name__)

# parameters that are contrast / orientation / polydispersity: fix by default,
# the data rarely constrains them and fitting them wrecks robustness.
_FIX_SUBSTR = ("sld", "polydisp")
_FIX_EXACT = {"theta", "phi", "psi", "up_frac_i", "up_frac_f", "up_theta"}
_FIX_SUFFIX = ("_pd", "_pd_n", "_pd_nsigma", "_pd_type")
# names that already carry the overall amplitude (so we needn't add 'scale')
_AMP_NAMES = ("i_zero", "lorentz_scale", "peak_scale", "porod_scale",
              "volfraction_a", "scale", "gauss_scale", "intensity")


def build_plan(model_name: str) -> dict:
    """A sensible default fit plan for ``model_name`` derived from its sasmodels
    parameters: fit the shape parameters + background (+ scale if no built-in
    amplitude), fix contrast/orientation/polydispersity, and take bounds from
    the model's own parameter limits where finite."""
    det = sascatalog.model_detail(model_name)
    params = (det or {}).get("parameters", []) or []
    fit, bounds, has_amp = [], {}, False
    for p in params:
        name = p["name"]
        ln = name.lower()
        if any(s in ln for s in _FIX_SUBSTR) or name in _FIX_EXACT \
                or ln.endswith(_FIX_SUFFIX):
            continue
        fit.append(name)
        if ln in _AMP_NAMES:
            has_amp = True
        lo, hi = (list(p.get("limits") or [None, None]) + [None, None])[:2]
        if _finite(lo) and _finite(hi) and hi > lo and abs(hi) < 1e30:
            bounds[name] = [float(lo), float(hi)]
    fit.append("background")
    if not has_amp:
        fit.append("scale")
    return {"model": model_name, "fit": fit, "bounds": bounds, "initial": {}}


def _fit_window(cand):
    """Match sas_agent's policy: if 'background' is fitted, keep the high-Q
    plateau (q_max=None) so the incoherent level is actually constrained."""
    qmin = (cand or {}).get("q_min")
    qmax = (cand or {}).get("q_max")
    if "background" in [p.lower() for p in ((cand or {}).get("fit") or [])]:
        qmax = None
    return qmin, qmax


def _fit(iq, cand, model_name, steps):
    qmin, qmax = _fit_window(cand)
    # sasmodels emits harmless overflow/divide warnings while the global search
    # probes extreme parameters; silence them so reportit logs stay readable.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return sasfit.fit_curve(
            iq.mod_q, iq.intensity, iq.error, model_name=model_name,
            initial=dict((cand or {}).get("initial") or {}),
            fit_params=(cand or {}).get("fit") or [],
            bounds=(cand or {}).get("bounds") or {},
            q_min=qmin, q_max=qmax, steps=steps)


def _score(r):
    if not r.ok:
        return {"score": 0.0}
    return shape.shape_score(r.q, r.i_data, r.i_model, reduced_chisq=r.reduced_chisq)


# Models that report a SIZE (Rg / radius) and therefore need a resolved Guinier
# region — a low-Q plateau — to be constrained. Discouraged when the data shows
# no such plateau (the fit then fails at low Q; use a correlation/OZ model).
GUINIER_DEPENDENT = {
    "guinier", "guinier_porod", "mono_gauss_coil", "poly_gauss_coil",
    "polymer_excl_volume", "unified_power_rg", "sphere", "polydisperse_sphere",
    "core_shell_sphere", "fuzzy_sphere", "ellipsoid", "cylinder", "star_polymer",
}


def physics_factors(result, data_desc, model_name) -> tuple[float, list]:
    """A multiplicative (0,1] penalty on a fit's raw shape score encoding SANS
    physics the raw curve-agreement can miss:

      * a size-based model used where the data has NO Guinier plateau;
      * a fit that only works by EXCLUDING the low-Q upturn and then diverging
        from it (extrapolated model peels away from the excluded markers);
      * a fitted parameter pinned at a bound or left unconstrained.

    Returns (factor, reasons)."""
    factor, reasons = 1.0, []
    name = (model_name or "").lower()
    if name in GUINIER_DEPENDENT and data_desc.get("has_guinier_region") is False:
        factor *= 0.6
        reasons.append("size-based model but no Guinier plateau in the data")
    cf, cr = _coverage_factor(result)
    factor *= cf
    reasons += cr
    pf, pr = _param_factor(result)
    factor *= pf
    reasons += pr
    return factor, reasons


def _coverage_factor(result) -> tuple[float, list]:
    """Penalise excluding the low-Q upturn and then diverging from it."""
    n_ex = len(result.q_excluded or [])
    n_in = len(result.q or [])
    total = n_ex + n_in
    if total == 0 or n_ex == 0:
        return 1.0, []
    reasons = []
    frac = n_ex / total
    div = 1.0
    try:
        qf = np.asarray(result.q_full, float)
        mf = np.asarray(result.i_model_full, float)
        qe = np.asarray(result.q_excluded, float)
        ie = np.asarray(result.i_excluded, float)
        me = (qe > 0) & (ie > 0) & np.isfinite(qe) & np.isfinite(ie)
        ok = (qf > 0) & (mf > 0) & np.isfinite(qf) & np.isfinite(mf)
        qe, ie = qe[me], ie[me]
        if qe.size >= 3 and ok.sum() >= 3:
            order = np.argsort(qf[ok])
            lm = np.interp(np.log10(qe), np.log10(qf[ok][order]), np.log10(mf[ok][order]))
            rms = float(np.sqrt(np.mean((np.log10(ie) - lm) ** 2)))
            div = float(np.exp(-max(0.0, rms - 0.05) / 0.2))
            if div < 0.9:
                reasons.append(f"model diverges from excluded data ({rms:.2f} dex off)")
    except Exception:  # noqa: BLE001
        pass
    if frac > 0.1:
        reasons.append(f"{100 * frac:.0f}% of points excluded from the fit")
    return div * (1.0 - min(frac, 0.5) * 0.8), reasons


def _param_factor(result) -> tuple[float, list]:
    """Penalise pinned-at-bound or unconstrained fitted parameters."""
    factor, reasons = 1.0, []
    if "at bound" in (result.note or "").lower():
        factor *= 0.7
        reasons.append("parameter pinned at a bound")
    for p, v in (result.params or {}).items():
        u = (result.uncertainties or {}).get(p)
        if u is None:
            continue
        if not np.isfinite(u):
            factor *= 0.75
            reasons.append(f"{p} unconstrained (non-finite uncertainty)")
            break
        if abs(v) > 0 and u / abs(v) > 2.0:
            factor *= 0.75
            reasons.append(f"{p} unconstrained (±{u:.2g} on {v:.2g})")
            break
    return factor, reasons


def _base_size(data_desc: dict) -> float:
    knee = data_desc.get("knee_q")
    if knee and knee > 0:
        return 1.0 / knee
    qmin, qmax = data_desc.get("q_min"), data_desc.get("q_max")
    if qmin and qmax and qmin > 0:
        return 1.0 / float(np.sqrt(qmin * qmax))     # geometric-mean Q
    return 50.0


def _sampled_initials(cand, data_desc, n_starts):
    """``n_starts`` diverse initial-value dicts.

    We deliberately vary only the parameters that create local minima — the
    SIZE parameter (swept log-uniformly around the data knee) and the power-law
    EXPONENTS (swept over their physical range) — and leave amplitude / scale /
    background to fit_curve's data-driven seeding so every start is anchored to
    the measured intensity level.
    """
    rng = np.random.default_rng(0)          # fixed => reproducible reruns
    fit_params = [p for p in (cand.get("fit") or [])]
    bounds = cand.get("bounds") or {}
    size_params = [p for p in fit_params if p.lower() in shape.SIZE_PARAMS]
    exp_params = [p for p in fit_params if p.lower() in shape.EXP_PARAMS]
    base_size = _base_size(data_desc)

    seeds = []
    for _ in range(max(0, n_starts)):
        init = {}
        for p in size_params:
            val = base_size * float(10 ** rng.uniform(-0.8, 0.8))  # ~0.15x..6x
            init[p] = _clamp(val, bounds.get(p))
        for p in exp_params:
            lo, hi = (bounds.get(p) or [1.0, 4.5])[:2]
            lo = max(lo, 0.5) if _finite(lo) else 0.5
            hi = min(hi, 6.0) if _finite(hi) else 6.0
            init[p] = float(rng.uniform(lo, min(hi, max(lo + 0.1, 4.5))))
        seeds.append(init)
    return seeds


def multistart_fit(iq, cand, model_name, data_desc, *, n_starts: int = 8,
                   steps: int = 200):
    """Fit ``model_name`` from many starts; return (best_result, best_score,
    attempts).  ``best_result`` is None if every start failed."""
    attempts = []
    best_r, best_s = None, {"score": -1.0}

    # start 0: the plan as given (fit_curve applies its own data-driven seeding)
    trials = [("data-seed", dict(cand.get("initial") or {}))]
    trials += [(f"sampled#{k+1}", s)
               for k, s in enumerate(_sampled_initials(cand, data_desc, n_starts))]

    for tag, init in trials:
        c = copy.deepcopy(cand)
        c["initial"] = {**(cand.get("initial") or {}), **init}
        try:
            r = _fit(iq, c, model_name, steps)
        except Exception as e:  # noqa: BLE001
            attempts.append({"start": tag, "ok": False, "note": f"exception: {e}"})
            continue
        sc = _score(r)
        base = sc.get("score", 0.0) if r.ok else 0.0
        factor, penalties = physics_factors(r, data_desc, model_name) if r.ok else (1.0, [])
        adj = base * factor
        # 'score' is the PHYSICS-ADJUSTED score used for ranking; 'base_score' is
        # the raw curve-agreement; 'penalties' explains any down-weighting.
        sc.update({"base_score": round(base, 4), "score": round(adj, 4),
                   "physics_factor": round(factor, 3), "penalties": penalties})
        attempts.append({
            "start": tag, "ok": bool(r.ok), "reduced_chisq": r.reduced_chisq,
            "score": round(adj, 4), "base_score": round(base, 4),
            "penalties": penalties, "params": dict(r.params), "note": r.note,
        })
        if r.ok and adj > best_s.get("score", -1.0):
            best_r, best_s = r, sc

    return best_r, best_s, attempts


# --------------------------------------------------------------------------- #
def _finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _clamp(v, bnd):
    if not bnd:
        return float(v)
    lo, hi = (list(bnd) + [None, None])[:2]
    if _finite(lo):
        v = max(v, float(lo))
    if _finite(hi):
        v = min(v, float(hi))
    return float(v)
