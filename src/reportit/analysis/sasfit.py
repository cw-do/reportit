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
    if trim_low_q:
        q, i, dy, _ = clean_low_q(q, i, dy)
    good = np.isfinite(q) & np.isfinite(i) & (q > 0)
    q, i = q[good], i[good]
    dy = dy[good] if dy is not None else np.sqrt(np.abs(i) + 1e-12)
    dy = np.where(dy > 0, dy, np.sqrt(np.abs(i) + 1e-12))

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

    # extract values + uncertainties
    for p in fitted:
        par = getattr(model, p)
        res.params[p] = float(par.value)
        res.uncertainties[p] = float(getattr(par, "stderr", 0.0) or 0.0)
    for p in valid:
        if p not in fitted:
            par = getattr(model, p, None)
            if par is not None and hasattr(par, "value"):
                res.fixed[p] = float(par.value)

    try:
        i_model = experiment.theory()
        res.i_model = [float(v) for v in i_model]
        res.q = [float(v) for v in q]
        res.i_data = [float(v) for v in i]
        res.r_squared = _r_squared(i, np.asarray(i_model))
    except Exception as e:  # noqa: BLE001
        logger.debug("theory eval failed: %s", e)

    try:
        res.reduced_chisq = float(problem.chisq())
    except Exception:  # noqa: BLE001
        res.reduced_chisq = None

    res.ok = True
    return res


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
