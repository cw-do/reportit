"""Lightweight quantitative metrics for a 1D I(Q) curve."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..models import DatasetAnalysis
from .loaders import load_iq

logger = logging.getLogger(__name__)


def _loglog_slope(q: np.ndarray, i: np.ndarray) -> float | None:
    mask = (q > 0) & (i > 0) & np.isfinite(q) & np.isfinite(i)
    if mask.sum() < 3:
        return None
    lx, ly = np.log10(q[mask]), np.log10(i[mask])
    try:
        return float(np.polyfit(lx, ly, 1)[0])
    except Exception:
        return None


def analyze(output_name: str, variant: str, iq_path: str | Path) -> DatasetAnalysis:
    da = DatasetAnalysis(output_name=output_name, variant=variant)
    try:
        iq = load_iq(iq_path)
    except Exception as e:  # noqa: BLE001
        da.flags.append(f"unreadable:{e}")
        return da

    q, i = np.asarray(iq.mod_q), np.asarray(iq.intensity)
    good = np.isfinite(q) & np.isfinite(i)
    q, i = q[good], i[good]
    if q.size == 0:
        da.flags.append("empty")
        return da

    da.n_points = int(q.size)
    da.q_min = float(np.min(q[q > 0])) if np.any(q > 0) else float(np.min(q))
    da.q_max = float(np.max(q))
    if np.any(i < 0):
        da.flags.append("negative_I")

    # low-Q = lowest third by Q, high-Q = highest third
    order = np.argsort(q)
    q, i = q[order], i[order]
    third = max(3, q.size // 3)
    da.low_q_slope = _loglog_slope(q[:third], i[:third])
    da.high_q_slope = _loglog_slope(q[-third:], i[-third:])
    return da
