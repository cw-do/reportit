"""Detect and drop the lowest-Q outlier points.

The 1-2 lowest-Q points in EQSANS reductions are frequently artifacts of an
insufficient beam-stop mask and sit well off the smooth trend. We drop a leading
point only when it deviates strongly (in log-log space) from a line fit to the
next several points.
"""

from __future__ import annotations

import numpy as np


def clean_low_q(q, i, err=None, max_drop: int = 2, log_thresh: float = 0.25):
    """Return (q, i, err, n_dropped) with leading low-Q outliers removed.

    Only positive-Q, positive-I points are considered. A leading point is dropped
    if |log10 I_obs - log10 I_predicted| > log_thresh, where the prediction comes
    from a linear log-log fit to the following ~5 points.
    """
    q = np.asarray(q, dtype=float)
    i = np.asarray(i, dtype=float)
    err = np.asarray(err, dtype=float) if err is not None else None

    order = np.argsort(q)
    q, i = q[order], i[order]
    if err is not None:
        err = err[order]

    dropped = 0
    while dropped < max_drop and len(q) - dropped > 7:
        head = dropped
        nxt = slice(head + 1, head + 6)
        qn, in_ = q[nxt], i[nxt]
        m = (qn > 0) & (in_ > 0)
        if m.sum() < 3 or q[head] <= 0 or i[head] <= 0:
            break
        try:
            slope, intercept = np.polyfit(np.log10(qn[m]), np.log10(in_[m]), 1)
        except Exception:
            break
        predicted = slope * np.log10(q[head]) + intercept
        if abs(np.log10(i[head]) - predicted) > log_thresh:
            dropped += 1
        else:
            break

    if dropped:
        q, i = q[dropped:], i[dropped:]
        if err is not None:
            err = err[dropped:]
    return q, i, err, dropped
