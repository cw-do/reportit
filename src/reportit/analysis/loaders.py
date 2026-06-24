"""Native numpy loaders for EQSANS reduced data — no drtsans dependency.

1D  *_Iq.dat       : tab-sep, 2 header lines, cols Q, I, dI, dQ
merged_*_Iq.txt    : tab-sep, header lines, cols Q, I, dI
2D  *_Iqxqy.dat    : tab-sep, 4 header lines incl. a literal "ASCII data" line
                     and a blank line, cols Qx, Qy, I, dI, dQx, dQy
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)


def _numeric_rows(path: Path) -> np.ndarray:
    """Read only rows whose first token parses as a float (skips text headers)."""
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                float(parts[0])
            except (ValueError, IndexError):
                continue  # e.g. the literal "ASCII data" line
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows:
        return np.empty((0, 0))
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows], dtype=float)


def load_iq(path: str | Path) -> SimpleNamespace:
    """Load a 1D I(Q) curve (handles both 4-col _Iq.dat and 3-col merged)."""
    data = _numeric_rows(Path(path))
    if data.size == 0:
        raise ValueError(f"No numeric data in {path}")
    return SimpleNamespace(
        mod_q=data[:, 0],
        intensity=data[:, 1],
        error=data[:, 2] if data.shape[1] > 2 else None,
        dq=data[:, 3] if data.shape[1] > 3 else None,
    )


def load_iqxqy(path: str | Path) -> SimpleNamespace:
    """Load a 2D I(Qx,Qy) dataset."""
    data = _numeric_rows(Path(path))
    if data.size == 0 or data.shape[1] < 3:
        raise ValueError(f"No 2D data in {path}")
    return SimpleNamespace(
        qx=data[:, 0],
        qy=data[:, 1],
        intensity=data[:, 2],
        error=data[:, 3] if data.shape[1] > 3 else None,
    )


def downsample_curve(path: str | Path, n: int = 30) -> list[tuple[float, float]]:
    """Return ~n (Q, I) points spanning the curve — for letting the LLM 'see' data."""
    iq = load_iq(path)
    q, i = iq.mod_q, iq.intensity
    if len(q) <= n:
        idx = range(len(q))
    else:
        idx = np.linspace(0, len(q) - 1, n).astype(int)
    return [(float(q[k]), float(i[k])) for k in idx]
