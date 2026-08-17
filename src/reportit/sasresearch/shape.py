"""Shape-aware fitness for 1D I(Q) fits.

A human judges a SANS fit by two things at once:

  * quantitatively — reduced chi^2 and the size of the residuals, and
  * qualitatively — does the model reproduce the *shape*?  Right curvature,
    the knee in the right place, a bump/valley where the data has one, and a
    high-Q slope that matches.

Pure reduced-chi^2 (what the old fitter ranked on) misses the second kind: a
low-chi^2 fit with a systematic S-shaped residual, or a knee in the wrong
place, looks bad to a human but can win on chi^2 alone.  This module turns the
qualitative judgement into numbers so the autoresearch loop can rank fits the
way a human would.

Everything is computed in log-log space, because that is the space the curves
are plotted and judged in.
"""

from __future__ import annotations

import numpy as np

# feature-name groups reused by the seeder (search.py) and by reporting
SIZE_PARAMS = ("rg", "cor_length", "correlation_length", "radius", "length",
               "xi", "d", "rg_1", "lorentz_length")
EXP_PARAMS = ("porod_exp", "lorentz_exp", "s", "m", "width_exp", "shape_exp",
              "exponent", "dim")


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def _clean(q, i):
    q = np.asarray(q, float)
    i = np.asarray(i, float)
    m = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    q, i = q[m], i[m]
    order = np.argsort(q)
    return q[order], i[order]


def local_slopes(q, i, nodes, win_dex: float = 0.22):
    """Local log-log slope d(lnI)/d(lnQ) at each node Q.

    A moving window ``win_dex`` decades wide is fit with a straight line in
    log-log space; that slope is the local power-law exponent a human reads off
    the curve.  Returns an array aligned with ``nodes`` (NaN where undefined).
    """
    q, i = _clean(q, i)
    if q.size < 3:
        return np.full(len(nodes), np.nan)
    lq, li = np.log10(q), np.log10(i)
    out = []
    for q0 in nodes:
        l0 = np.log10(q0)
        sel = np.abs(lq - l0) <= win_dex
        if sel.sum() < 3:  # widen: take the 5 nearest points
            sel = np.argsort(np.abs(lq - l0))[:5]
        try:
            out.append(float(np.polyfit(lq[sel], li[sel], 1)[0]))
        except Exception:  # noqa: BLE001
            out.append(np.nan)
    return np.asarray(out, float)


def slope_nodes(q, n: int = 14):
    """``n`` log-spaced Q nodes spanning the (positive) data range."""
    q = np.asarray(q, float)
    q = q[np.isfinite(q) & (q > 0)]
    if q.size < 2:
        return np.asarray([], float)
    return np.logspace(np.log10(q.min()), np.log10(q.max()), n)


def _smooth(y, k: int = 3):
    if k < 2 or y.size < k:
        return y
    ker = np.ones(k) / k
    return np.convolve(y, ker, mode="same")


def find_features(q, i, *, min_prom: float = 0.04):
    """Detect bumps (local maxima) and valleys (local minima) in log-log I(Q).

    ``min_prom`` is the minimum prominence in decades of I for a feature to
    count — filters out noise wiggles.  Returns ({"peaks": [...], "valleys":
    [...]}) as lists of Q positions, low-Q first.
    """
    q, i = _clean(q, i)
    peaks, valleys = [], []
    if q.size < 7:
        return {"peaks": peaks, "valleys": valleys}
    li = _smooth(np.log10(i), 3)
    for k in range(1, len(li) - 1):
        left, right = li[k] - li[k - 1], li[k] - li[k + 1]
        if left > 0 and right > 0:                       # local max
            prom = min(li[k] - li[:k].min(), li[k] - li[k:].min())
            if prom >= min_prom:
                peaks.append(float(q[k]))
        elif left < 0 and right < 0:                     # local min
            prom = min(li[:k].max() - li[k], li[k:].max() - li[k])
            if prom >= min_prom:
                valleys.append(float(q[k]))
    return {"peaks": peaks, "valleys": valleys}


def find_knee(q, i):
    """Q of the sharpest downward bend (the Guinier / correlation knee): the
    node where the local slope drops most steeply.  None if not well defined."""
    nodes = slope_nodes(q, 16)
    if nodes.size < 4:
        return None
    s = local_slopes(q, i, nodes)
    ds = np.diff(s)
    ds = np.where(np.isfinite(ds), ds, 0.0)
    if not ds.size or ds.min() >= -0.15:   # no meaningful bend
        return None
    return float(nodes[int(np.argmin(ds)) + 1])


def has_guinier_region(q, i, *, flat_thresh: float = -0.4) -> bool | None:
    """Whether the curve shows a low-Q Guinier region — a plateau where I(Q)
    FLATTENS (log-log slope approaching 0) within the measured Q-range.

    This is the make-or-break condition for size-based models (guinier,
    guinier_porod, mono/poly_gauss_coil, polymer_excl_volume, form factors):
    they can only pin a size (Rg) if the size scale is resolved, i.e. the curve
    bends over to a plateau at low Q. If instead the low-Q keeps RISING (an
    upturn or an unbroken power law with no flattening), the size is
    unconstrained and such a model fails at low Q — a correlation-length /
    Lorentzian description (which needs no Guinier plateau) is appropriate.

    Returns True if any node in the low-Q ~40% has a near-flat slope
    (> flat_thresh, i.e. between ~-0.4 and 0), else False; None if undecidable.
    Note: a Lorentzian/OZ curve also flattens at low Q, so True does NOT by
    itself favour a size model — it only means size models are *admissible*; the
    score still decides. False is the strong signal (size models discouraged).
    """
    q, i = _clean(q, i)
    if q.size < 8:
        return None
    nodes = slope_nodes(q, 14)
    s = local_slopes(q, i, nodes)
    good = np.isfinite(s)
    if good.sum() < 4:
        return None
    nodes, s = nodes[good], s[good]
    q40 = 10 ** (np.log10(nodes.min()) + 0.4 * (np.log10(nodes.max()) - np.log10(nodes.min())))
    low = s[nodes <= q40]
    if low.size == 0:
        low = s[:2]
    return bool(np.any(low > flat_thresh))


def curve_descriptors(q, i) -> dict:
    """Human-readable shape summary of one curve (for the LLM and the notebook)."""
    q, i = _clean(q, i)
    if q.size < 3:
        return {"n_points": int(q.size)}
    nodes = slope_nodes(q, 14)
    s = local_slopes(q, i, nodes)
    third = max(3, q.size // 3)
    feats = find_features(q, i)
    return {
        "n_points": int(q.size),
        "q_min": float(q.min()), "q_max": float(q.max()),
        "low_q_slope": _safe(np.polyfit(np.log10(q[:third]), np.log10(i[:third]), 1)[0]),
        "high_q_slope": _safe(np.polyfit(np.log10(q[-third:]), np.log10(i[-third:]), 1)[0]),
        "slope_profile": [[round(float(x), 5), round(float(y), 3)]
                          for x, y in zip(nodes, s) if np.isfinite(y)],
        "knee_q": find_knee(q, i),
        "peaks_q": feats["peaks"],
        "valleys_q": feats["valleys"],
        "low_q_plateau": bool(np.isfinite(s[0]) and abs(s[0]) < 0.35),
        "has_guinier_region": has_guinier_region(q, i),
    }


def residual_runs_z(residuals) -> float:
    """Wald-Wolfowitz runs-test z on residual signs (residuals ordered by Q).

    Random residuals alternate sign ~as expected; a systematic misfit produces
    long same-sign streaks => FEWER runs than expected => NEGATIVE z.  We only
    penalise the negative side.
    """
    r = np.asarray(residuals, float)
    r = r[np.isfinite(r) & (r != 0)]
    n = r.size
    if n < 8:
        return 0.0
    signs = r > 0
    n_pos = int(signs.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:            # all one side: maximally systematic
        return -abs(n)
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mu = 1 + 2 * n_pos * n_neg / n
    var = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n * n * (n - 1))
    if var <= 0:
        return 0.0
    return float((runs - mu) / np.sqrt(var))


# --------------------------------------------------------------------------- #
# the composite score
# --------------------------------------------------------------------------- #
def _match_offset(a: list, b: list) -> float | None:
    """Mean nearest-neighbour distance (in decades of Q) from features ``a`` to
    features ``b``.  Missing counterparts cost 0.5 dex.  None if ``a`` empty."""
    if not a:
        return None
    if not b:
        return 0.5
    la, lb = np.log10(np.asarray(a, float)), np.log10(np.asarray(b, float))
    return float(np.mean([min(abs(x - lb)) for x in la]))


def shape_score(q, i_data, i_model, *, reduced_chisq: float | None = None) -> dict:
    """Score how well ``i_model`` reproduces ``i_data`` over ``q``.

    Returns a dict of sub-scores in [0, 1] (1 = perfect) plus a composite
    ``score``.  Higher is better throughout.  Sub-scores:

      resid   — size of the log-log residual (the vertical gap human sees)
      chisq   — reduced chi^2 term (skipped/redistributed if unavailable)
      slope   — agreement of the local slope profile (curvature everywhere)
      feature — agreement of knee / bump / valley positions
      runs    — residual randomness (systematic streaks penalised)
    """
    q = np.asarray(q, float)
    yd = np.asarray(i_data, float)
    ym = np.asarray(i_model, float)
    m = np.isfinite(q) & np.isfinite(yd) & np.isfinite(ym) & (q > 0) & (yd > 0) & (ym > 0)
    out = {"resid": 0.0, "chisq": 0.0, "slope": 0.0, "feature": 0.0,
           "runs": 0.0, "score": 0.0, "n": int(m.sum())}
    if m.sum() < 5:
        return out
    q, yd, ym = q[m], yd[m], ym[m]
    order = np.argsort(q)
    q, yd, ym = q[order], yd[order], ym[order]

    # 1) log-log residual size (a human's "vertical gap"): 0.15 dex ~ 40% => ~0.37
    r = np.log10(yd) - np.log10(ym)
    log_rms = float(np.sqrt(np.mean(r ** 2)))
    out["resid"] = float(np.exp(-log_rms / 0.15))

    # 2) reduced chi^2 (from the fitter's error bars): 1 is ideal
    if reduced_chisq is not None and np.isfinite(reduced_chisq):
        out["chisq"] = float(1.0 / (1.0 + max(0.0, reduced_chisq - 1.0)))
        have_chisq = True
    else:
        have_chisq = False

    # 3) slope-profile agreement (does the curvature track everywhere?)
    nodes = slope_nodes(q, 14)
    sd = local_slopes(q, yd, nodes)
    sm = local_slopes(q, ym, nodes)
    good = np.isfinite(sd) & np.isfinite(sm)
    if good.sum() >= 3:
        slope_rms = float(np.sqrt(np.mean((sd[good] - sm[good]) ** 2)))
        out["slope"] = float(np.exp(-slope_rms / 0.5))   # 0.5 in exponent => forgiving
    else:
        out["slope"] = 0.5

    # 4) feature-position agreement (knee / bump / valley in the same place?)
    fd, fm = find_features(q, yd), find_features(q, ym)
    kd, km = find_knee(q, yd), find_knee(q, ym)
    offs = [o for o in (
        _match_offset(fd["peaks"], fm["peaks"]),
        _match_offset(fd["valleys"], fm["valleys"]),
        _match_offset([kd] if kd else [], [km] if km else []),
    ) if o is not None]
    out["feature"] = float(np.exp(-np.mean(offs) / 0.15)) if offs else 1.0

    # 5) residual randomness
    z = residual_runs_z(r)
    out["runs"] = float(1.0 / (1.0 + max(0.0, -z) / 2.0))

    # composite — resid + curvature dominate; chi^2 is a supporting witness
    if have_chisq:
        w = {"resid": 0.32, "chisq": 0.18, "slope": 0.27, "feature": 0.11, "runs": 0.12}
    else:
        w = {"resid": 0.42, "chisq": 0.0, "slope": 0.32, "feature": 0.13, "runs": 0.13}
    out["score"] = float(sum(w[k] * out[k] for k in w))
    return out


def _safe(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:  # noqa: BLE001
        return None
