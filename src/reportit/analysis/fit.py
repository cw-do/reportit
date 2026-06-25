"""Optional model fits, run only when the LLM strategy requests them.

Supported: Guinier (Rg, I0), power-law / Porod (exponent, prefactor).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import FitResult
from .clean import clean_low_q
from .loaders import load_iq

logger = logging.getLogger(__name__)


def _r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _window(q, i, qmin, qmax):
    m = np.isfinite(q) & np.isfinite(i) & (i > 0) & (q > 0)
    if qmin is not None:
        m &= q >= qmin
    if qmax is not None:
        m &= q <= qmax
    return q[m], i[m]


def guinier_fit(q, i, qmin=None, qmax=None) -> FitResult:
    """ln I = ln I0 - (Rg^2/3) q^2  → linear fit of ln(I) vs q^2."""
    res = FitResult(kind="guinier", q_range=(qmin or 0.0, qmax or 0.0))
    q, i = _window(np.asarray(q), np.asarray(i), qmin, qmax)
    if q.size < 4:
        res.note = "too few points"
        return res
    x, y = q ** 2, np.log(i)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception as e:  # noqa: BLE001
        res.note = str(e)
        return res
    if slope >= 0:
        res.note = "non-physical (slope>=0)"
        return res
    rg = float(np.sqrt(-3.0 * slope))
    i0 = float(np.exp(intercept))
    res.params = {"Rg": rg, "I0": i0}
    res.q_range = (float(q.min()), float(q.max()))
    res.r_squared = _r_squared(y, slope * x + intercept)
    res.ok = True
    return res


def powerlaw_fit(q, i, qmin=None, qmax=None, kind="powerlaw") -> FitResult:
    """I = A q^p  → linear fit of log10(I) vs log10(q)."""
    res = FitResult(kind=kind, q_range=(qmin or 0.0, qmax or 0.0))
    q, i = _window(np.asarray(q), np.asarray(i), qmin, qmax)
    if q.size < 4:
        res.note = "too few points"
        return res
    x, y = np.log10(q), np.log10(i)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception as e:  # noqa: BLE001
        res.note = str(e)
        return res
    res.params = {"exponent": float(slope), "prefactor": float(10 ** intercept)}
    res.q_range = (float(q.min()), float(q.max()))
    res.r_squared = _r_squared(y, slope * x + intercept)
    res.ok = True
    return res


def correlation_fit(q, i, err=None, qmin=None, qmax=None) -> FitResult:
    """Ornstein-Zernike correlation-length fit: I(q) = I0 / (1 + (q*xi)^2) + bkg.

    Good for solution scattering with a low-Q plateau rolling into a power law —
    the regime where Guinier (compact-particle) analysis is inappropriate.
    """
    from scipy.optimize import curve_fit

    res = FitResult(kind="correlation", q_range=(qmin or 0.0, qmax or 0.0))
    q, i = _window(np.asarray(q, float), np.asarray(i, float), qmin, qmax)
    if q.size < 5:
        res.note = "too few points"
        return res

    def model(qq, i0, xi, bkg):
        return i0 / (1.0 + (qq * xi) ** 2) + bkg

    i0_0 = float(np.max(i))
    bkg_0 = float(max(np.min(i), 0.0))
    # xi guess: where intensity falls to half of its low-q value
    half = i0_0 / 2.0
    below = q[i <= half]
    xi_0 = float(1.0 / below[0]) if below.size else float(1.0 / np.median(q))
    try:
        popt, _ = curve_fit(model, q, i, p0=[i0_0, xi_0, bkg_0],
                            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=10000)
    except Exception as e:  # noqa: BLE001
        res.note = str(e)
        return res
    i0, xi, bkg = (float(v) for v in popt)
    yhat = model(q, *popt)
    res.params = {"I0": i0, "xi": xi, "bkg": bkg}
    res.q_range = (float(q.min()), float(q.max()))
    res.r_squared = _r_squared(i, yhat)
    res.ok = True
    return res


def run_fit(iq_path: str | Path, model: str, qmin: Optional[float] = None,
            qmax: Optional[float] = None, trim_low_q: bool = True) -> FitResult:
    iq = load_iq(iq_path)
    q, i, err = np.asarray(iq.mod_q), np.asarray(iq.intensity), iq.error
    if trim_low_q:
        q, i, err, _ = clean_low_q(q, i, err)
    model = (model or "").lower()
    if model == "guinier":
        return guinier_fit(q, i, qmin, qmax)
    if model in ("correlation", "ornstein_zernike", "ornstein-zernike", "oz", "lorentzian"):
        return correlation_fit(q, i, err, qmin, qmax)
    if model in ("porod", "powerlaw", "power-law", "power_law"):
        return powerlaw_fit(q, i, qmin, qmax,
                            kind="porod" if model == "porod" else "powerlaw")
    return FitResult(kind=model or "unknown", note="unsupported model")
