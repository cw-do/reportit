"""Optional model fits, run only when the LLM strategy requests them.

Supported: Guinier (Rg, I0), power-law / Porod (exponent, prefactor).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import FitResult
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


def run_fit(iq_path: str | Path, model: str, qmin: Optional[float] = None,
            qmax: Optional[float] = None) -> FitResult:
    iq = load_iq(iq_path)
    model = (model or "").lower()
    if model == "guinier":
        return guinier_fit(iq.mod_q, iq.intensity, qmin, qmax)
    if model in ("porod", "powerlaw", "power-law", "power_law"):
        return powerlaw_fit(iq.mod_q, iq.intensity, qmin, qmax,
                            kind="porod" if model == "porod" else "powerlaw")
    return FitResult(kind=model or "unknown", note="unsupported model")
