"""Fit a 1D I(Q) curve to a sasmodels model with bumps.

Mirrors SasAgent's SAS/fitting.py approach (load_model -> bumps_model.Model /
Experiment -> FitProblem -> bumps.fitters.fit), adapted to take a parameter plan
(initial values, which parameters to fit + bounds, which to keep fixed).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..models import SasFitResult
from .clean import clean_low_q

logger = logging.getLogger(__name__)


def fit_curve(
    q, i, dy=None, *,
    model_name: str,
    initial: Optional[dict] = None,
    fit_params: Optional[list] = None,
    bounds: Optional[dict] = None,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
    trim_low_q: bool = True,
    drop_lowest: int = 1,
    steps: int = 300,
) -> SasFitResult:
    """Fit (q, i) to `model_name`, optionally over a restricted [q_min, q_max].

    Restricting the window is a first-class feature: a model may validly describe
    only part of the curve (e.g. exclude a low-Q aggregation upturn that lies
    outside the length scale of interest). Excluded points are retained for
    plotting/context.

    initial:    {param: value} starting guesses (others use model defaults)
    fit_params: list of parameters to optimize (others stay fixed)
    bounds:     {param: [lo, hi]} optimization bounds for fitted parameters
    """
    initial = initial or {}
    fit_params = fit_params or []
    bounds = bounds or {}

    res = SasFitResult(model_name=model_name)
    try:
        from bumps.fitters import fit as bumps_fit
        from bumps.fitproblem import FitProblem
        from sasmodels.bumps_model import Experiment, Model
        from sasmodels.core import load_model
        from sasmodels.data import Data1D
    except Exception as e:  # noqa: BLE001
        res.note = f"sasmodels/bumps import failed: {e}"
        return res

    q = np.asarray(q, float)
    i = np.asarray(i, float)
    dy = np.asarray(dy, float) if dy is not None else None
    good = np.isfinite(q) & np.isfinite(i) & (q > 0)
    q, i = q[good], i[good]
    dy = dy[good] if dy is not None else dy
    # drop the very lowest-Q point(s): the beam-stop/mask region is frequently an
    # artifact in EQSANS merged data (general, not sample-specific).
    if drop_lowest > 0 and q.size > drop_lowest + 6:
        order = np.argsort(q)
        q, i = q[order], i[order]
        dy = dy[order] if dy is not None else None
        q, i = q[drop_lowest:], i[drop_lowest:]
        dy = dy[drop_lowest:] if dy is not None else None
    if trim_low_q:
        q, i, dy, _ = clean_low_q(q, i, dy)
    good = np.isfinite(q) & np.isfinite(i) & (q > 0)
    q, i = q[good], i[good]
    dy = dy[good] if dy is not None else np.sqrt(np.abs(i) + 1e-12)
    dy = np.where(dy > 0, dy, np.sqrt(np.abs(i) + 1e-12))
    order = np.argsort(q)
    q, i, dy = q[order], i[order], dy[order]
    q_all, i_all = q.copy(), i.copy()  # full curve (for plotting the model beyond the fit window)

    # restrict to the requested fit window; keep excluded points for context
    in_win = np.ones(q.shape, dtype=bool)
    if q_min is not None:
        in_win &= q >= float(q_min)
    if q_max is not None:
        in_win &= q <= float(q_max)
    if in_win.sum() >= 6:
        res.q_excluded = [float(v) for v in q[~in_win]]
        res.i_excluded = [float(v) for v in i[~in_win]]
        res.fit_qmin = float(q[in_win].min())
        res.fit_qmax = float(q[in_win].max())
        q, i, dy = q[in_win], i[in_win], dy[in_win]
    if q.size < 6:
        res.note = "too few points in fit window"
        return res

    try:
        kernel = load_model(model_name)
    except Exception as e:  # noqa: BLE001
        res.note = f"unknown model {model_name!r}: {e}"
        return res

    valid = {p.name for p in kernel.info.parameters.kernel_parameters}
    valid |= {"scale", "background"}
    init = {k: v for k, v in initial.items() if k in valid}
    # Data-driven incoherent-background initial guess: match the high-Q plateau
    # BEFORE fitting (correlation-length & similar fits are very sensitive to it).
    bounds = dict(bounds)
    bkg0 = _estimate_background(q_all, i_all) if "background" in valid else None
    if bkg0 is not None and bkg0 > 0:
        init["background"] = bkg0
        # anchor the background near the measured high-Q plateau: tight,
        # data-driven bounds (overriding any LLM guess) so the fit locks the
        # incoherent level instead of letting other parameters compensate.
        bounds["background"] = [0.3 * bkg0, 2.0 * bkg0]
    # seed the amplitude/size parameters from the data so weak signals on a high
    # background are not collapsed by a far-off start (general, not per-sample).
    _seed_from_data(q, i, bkg0, fit_params, valid, init)
    try:
        model = Model(kernel, **init)
    except Exception as e:  # noqa: BLE001
        res.note = f"bad initial params: {e}"
        return res

    # mark fitted parameters with a range; the rest stay fixed
    fitted = []
    for p in fit_params:
        if p not in valid:
            continue
        lo, hi = (bounds.get(p) or [None, None])[:2]
        par = getattr(model, p, None)
        if par is None:
            continue
        try:
            if lo is not None and hi is not None and hi > lo:
                par.range(float(lo), float(hi))
            else:
                par.range(*_default_range(par))
            fitted.append(p)
        except Exception as e:  # noqa: BLE001
            logger.debug("range set failed for %s: %s", p, e)
    if not fitted:
        res.note = "no fittable parameters"
        return res

    data = Data1D(x=q, y=i, dy=dy)
    try:
        experiment = Experiment(data=data, model=model)
        problem = FitProblem(experiment)
        # Stage 1: fast local fit.
        bumps_fit(problem, method="lm", steps=steps, verbose=False)
        chisq1 = _safe_chisq(problem)
        snap1 = {p: getattr(model, p).value for p in fitted}
        # Stage 2: if the local fit is poor or a parameter is pinned at a bound,
        # it is likely stuck in a local minimum — do a GLOBAL search (differential
        # evolution) then refine locally, and keep whichever is better.
        if _poor_fit(chisq1, model, fitted):
            try:
                bumps_fit(problem, method="de", steps=max(steps, 200), verbose=False)
                bumps_fit(problem, method="lm", steps=steps, verbose=False)
                chisq2 = _safe_chisq(problem)
                if not (chisq2 < chisq1):  # global+refine not better -> restore local
                    for p, v in snap1.items():
                        getattr(model, p).value = v
                    res.note = "global search did not improve local fit"
                else:
                    res.note = "global search (DE) used to escape local minimum"
            except Exception as e:  # noqa: BLE001
                logger.debug("global fit stage failed: %s", e)
                for p, v in snap1.items():
                    getattr(model, p).value = v
    except Exception as e:  # noqa: BLE001
        res.note = f"fit failed: {e}"
        return res

    # extract values
    for p in fitted:
        res.params[p] = float(getattr(model, p).value)
    for p in valid:
        if p not in fitted:
            par = getattr(model, p, None)
            if par is not None and hasattr(par, "value"):
                res.fixed[p] = float(par.value)

    # parameter uncertainties from the fit covariance (lm/de don't fill stderr) +
    # flag any parameter pinned at a bound (a sign the fit is unreliable).
    try:
        cov = problem.cov(problem.getp())
        errs = np.sqrt(np.abs(np.diag(np.asarray(cov, float))))
        labels = list(problem.labels())
        for lab, se in zip(labels, errs):
            key = lab.split(".")[-1]
            if key in res.params and np.isfinite(se):
                res.uncertainties[key] = float(se)
    except Exception as e:  # noqa: BLE001
        logger.debug("covariance/uncertainty estimate failed: %s", e)
    pinned = _pinned_params(model, fitted)
    if pinned:
        res.note = (res.note + "; " if res.note else "") + \
            "parameter(s) at bound: " + ", ".join(pinned)

    try:
        i_model = experiment.theory()
        res.i_model = [float(v) for v in i_model]
        res.q = [float(v) for v in q]
        res.i_data = [float(v) for v in i]
        res.r_squared = _r_squared(i, np.asarray(i_model))
    except Exception as e:  # noqa: BLE001
        logger.debug("theory eval failed: %s", e)

    # evaluate the fitted model over the FULL Q range so the report can show the
    # model extended (dashed) beyond the fitted window.
    try:
        from sasmodels.direct_model import DirectModel
        allp = {**res.fixed, **res.params}
        dm = DirectModel(Data1D(x=q_all, y=np.ones_like(q_all),
                                dy=np.ones_like(q_all)), kernel)
        i_full = dm(**allp)
        res.q_full = [float(v) for v in q_all]
        res.i_model_full = [float(v) for v in i_full]
    except Exception as e:  # noqa: BLE001
        logger.debug("full-range model eval failed: %s", e)

    try:
        res.reduced_chisq = float(problem.chisq())
    except Exception:  # noqa: BLE001
        res.reduced_chisq = None

    res.ok = True
    return res


_AMP_PARAMS = ("i_zero", "lorentz_scale", "scale", "gauss_scale", "intensity")
_SIZE_PARAMS = ("rg", "cor_length", "correlation_length", "radius", "length", "rg_1")


def _seed_from_data(q, i, bkg, fit_params, valid, init):
    """Data-driven starting guesses for amplitude- and size-like parameters:
    amplitude ≈ low-Q excess above background; size ≈ 1/Q_knee (where the excess
    falls to half). Only used as a starting point; the fit still refines."""
    q = np.asarray(q, float)
    i = np.asarray(i, float)
    m = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    q, i = q[m], i[m]
    if q.size < 6:
        return
    order = np.argsort(q)
    q, i = q[order], i[order]
    b = bkg if (bkg is not None and bkg > 0) else float(np.median(i[int(0.8 * len(i)):]))
    excess = float(max(np.median(i[:max(3, len(i) // 20)]) - b, 1e-4))
    half = b + excess / 2.0
    below = q[i <= half]
    knee = float(below[0]) if below.size else float(q[len(q) // 3])
    size = float(1.0 / knee) if knee > 0 else None
    for p in fit_params:
        if p not in valid or p in init:   # only fill what the caller didn't set
            continue
        pl = p.lower()
        if pl in _AMP_PARAMS:
            init[p] = excess
        elif pl in _SIZE_PARAMS and size is not None:
            init[p] = size


def _pinned_params(model, fitted, eps: float = 1e-3) -> list:
    """Names of fitted parameters sitting at (within eps of) a bound."""
    out = []
    for p in fitted:
        par = getattr(model, p, None)
        try:
            lo, hi = par.bounds.limits
            v = par.value
            if hi > lo and (abs(v - lo) <= eps * (hi - lo) or abs(hi - v) <= eps * (hi - lo)):
                out.append(p)
        except Exception:  # noqa: BLE001
            continue
    return out


def _estimate_background(q, i):
    """Estimate the flat incoherent background from the high-Q plateau.

    Select by Q VALUE (the top of the Q range), not by point count — SANS data is
    dense at low Q, so a count-based quantile would bleed into mid-Q and badly
    overestimate. The incoherent level is the asymptotic floor at the highest Q.
    """
    q = np.asarray(q, float)
    i = np.asarray(i, float)
    m = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    q, i = q[m], i[m]
    if q.size < 8:
        return None
    qmax = q.max()
    hi = q >= 0.8 * qmax                   # top 20% of the Q RANGE by value
    if hi.sum() < 3:
        hi = q >= 0.6 * qmax
    vals = i[hi]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(max(np.median(vals), 0.0))


def _safe_chisq(problem):
    try:
        return float(problem.chisq())
    except Exception:  # noqa: BLE001
        return float("inf")


def _poor_fit(chisq, model, fitted, *, chisq_thresh: float = 2.0, eps: float = 1e-3) -> bool:
    """A fit is 'poor' if reduced chi^2 is high or a fitted parameter is pinned to
    a bound (a classic sign of a local-minimum / bad-start failure)."""
    if chisq is None or not np.isfinite(chisq) or chisq > chisq_thresh:
        return True
    for p in fitted:
        par = getattr(model, p, None)
        try:
            lo, hi = par.bounds.limits
            v = par.value
            if hi > lo and (abs(v - lo) <= eps * (hi - lo) or abs(hi - v) <= eps * (hi - lo)):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _default_range(par):
    lo, hi = getattr(par, "limits", (0.0, np.inf))
    val = float(getattr(par, "value", 1.0) or 1.0)
    lo = lo if (lo is not None and np.isfinite(lo)) else max(0.0, val * 0.01)
    hi = hi if (hi is not None and np.isfinite(hi)) else max(val * 100, 1.0)
    return float(lo), float(hi)


def _r_squared(y, yhat) -> Optional[float]:
    m = np.isfinite(y) & np.isfinite(yhat) & (y > 0)
    if m.sum() < 3:
        return None
    ly, lh = np.log10(y[m]), np.log10(np.clip(yhat[m], 1e-30, None))
    ss_res = float(np.sum((ly - lh) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None
