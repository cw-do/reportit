"""Repeat distance / d-spacing analysis.

Many SANS proposals are ultimately about a *periodicity*: a lamellar repeat, a
granum thylakoid stacking distance, an interlayer spacing. Such an experiment is
answered by one number per sample — the position of the correlation/Bragg peak,
converted to a real-space distance

    d = 2 pi / Q_peak

When the proposal says so, this module makes the report answer that question
directly instead of leaving the reader to convert Q by hand:

  * :func:`detect` decides, from the proposal's goals and hypotheses, whether a
    repeat distance is what the experiment is after (and picks up any Q or d
    range the proposal states);
  * :func:`find_peak` locates the peak in a measured curve WITHOUT a model, so a
    d-spacing is available even for groups that are not fitted;
  * :func:`from_fit` reads the peak position out of a fitted sasmodels result
    (broad_peak, gaussian_peak, lamellar, ...) — the better estimate where it
    exists, because the fit separates the peak from the decaying background and
    carries an uncertainty.

Both estimates are reported when both exist: they answer the same question by
different means, and a disagreement is information, not something to hide.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Wording that means "this experiment is about a periodicity".
#
# Deliberately SPECIFIC phrases only. Bare words like "repeat", "periodic", "RD"
# or "stacking" appear in almost any proposal ("repeat the measurement", "RD" as
# initials), and under --no-llm the proposal summary can be raw PDF text — a
# loose list then fires on every experiment. Each term here has to mean a
# structural periodicity on its own. 'repeat unit' is excluded too: in polymer
# science it means the monomer, not a spacing.
_TERMS = (
    r"repeat distance", r"repeat spacing", r"d-?spacing",
    r"\blamellar\b", r"periodicity", r"interlayer", r"layer spacing",
    r"bragg (?:peak|reflection|spacing)", r"lattice spacing",
    r"\bgranum\b", r"\bgrana\b", r"thylakoid",
    r"correlation peak", r"diffraction peak",
)
_TERM_RE = re.compile("|".join(_TERMS), re.IGNORECASE)

# Fitted parameters that ARE a peak position in Q (per sasmodels).
PEAK_Q_PARAMS = ("peak_pos", "q0", "peak_q")
# Fitted parameters that are ALREADY a real-space repeat distance.
PEAK_D_PARAMS = ("d_spacing", "spacing")

# Models worth trying when a repeat distance is the goal.
PEAK_MODELS = ("broad_peak", "gaussian_peak", "correlation_length", "teubner_strey",
               "lamellar", "lamellar_hg")


@dataclass
class Intent:
    """Whether the experiment targets a repeat distance, and any stated range."""

    wanted: bool = False
    terms: list[str] = field(default_factory=list)
    q_window: tuple[float, float] | None = None   # search window in 1/A
    source: str = ""                              # where the window came from

    @property
    def d_window(self) -> tuple[float, float] | None:
        if not self.q_window:
            return None
        lo, hi = self.q_window
        return (2 * math.pi / hi, 2 * math.pi / lo)


@dataclass
class Peak:
    q: float
    d: float
    snr: float
    method: str = "model-free"
    d_err: float | None = None


def d_of_q(q: float) -> float:
    """Real-space repeat distance (A) from a peak position in Q (1/A)."""
    return 2.0 * math.pi / float(q)


# --------------------------------------------------------------------------- #
# 1) does the proposal ask for a repeat distance?
# --------------------------------------------------------------------------- #
def detect(proposal) -> Intent:
    """Decide from the proposal whether a repeat distance is the goal."""
    if proposal is None or not getattr(proposal, "available", False):
        return Intent()
    chunks: list[str] = []
    for attr in ("title", "abstract_summary"):
        v = getattr(proposal, attr, None)
        if v:
            chunks.append(str(v))
    chunks += [str(g) for g in (getattr(proposal, "science_goals", None) or [])]
    for h in (getattr(proposal, "hypotheses", None) or []):
        chunks.append(str(getattr(h, "text", "")))
        chunks.append(str(getattr(h, "expected_signature", "")))
    blob = "\n".join(c for c in chunks if c)
    hits = sorted({m.group(0).lower() for m in _TERM_RE.finditer(blob)})
    if not hits:
        return Intent()
    q_window, source = _stated_window(blob)
    return Intent(wanted=True, terms=hits, q_window=q_window, source=source)


_Q_RANGE_RE = re.compile(
    r"Q[^.\n]{0,40}?(\d*\.\d+)\s*(?:-|–|—|to)\s*(\d*\.\d+)\s*(?:A|Å|1/A)?",
    re.IGNORECASE)
_D_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(nm|A|Å)\b", re.IGNORECASE)


def _stated_window(blob: str) -> tuple[tuple[float, float] | None, str]:
    """A Q (or d) range the proposal states, converted to a Q window in 1/A."""
    m = _Q_RANGE_RE.search(blob)
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        if 1e-4 < lo < hi < 1.0:
            return (lo, hi), "Q-range stated in the proposal"
    m = _D_RANGE_RE.search(blob)
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        scale = 10.0 if m.group(3).lower() == "nm" else 1.0
        lo, hi = lo * scale, hi * scale
        if 5.0 < lo < hi < 10000.0:
            return (2 * math.pi / hi, 2 * math.pi / lo), "d-range stated in the proposal"
    return None, ""


# --------------------------------------------------------------------------- #
# 2) model-free peak position
# --------------------------------------------------------------------------- #
def default_window(q) -> tuple[float, float]:
    """A sensible Q band to look for a repeat-distance peak in, when the proposal
    states none.

    The top of the measured range is incoherent-background dominated, where noise
    wiggles can out-score a real correlation peak, and the first points carry
    beam-stop artefacts. Searching there produces confident nonsense (a 'peak' at
    Q~0.4 is a 1.5 nm 'repeat distance'), so the search is confined to the lower
    ~2/3 of the measured range in log Q.
    """
    q = np.asarray(q, float)
    q = q[np.isfinite(q) & (q > 0)]
    if q.size < 10:
        return (0.0, float("inf"))
    lo, hi = float(q.min()), float(q.max())
    llo, lhi = np.log10(lo), np.log10(hi)
    return (10 ** (llo + 0.10 * (lhi - llo)), 10 ** (llo + 0.65 * (lhi - llo)))


def find_peak(q, i, q_window: tuple[float, float] | None = None, *,
              min_snr: float = 3.0, baseline_deg: int = 4) -> Peak | None:
    """Locate a correlation peak without fitting a model. None if none is clear.

    A correlation peak in SANS usually rides on a steeply decaying background, so
    the raw curve rarely has a local maximum where the peak is. The background is
    therefore removed first — a low-order polynomial in log-log, which absorbs
    both the power-law decay and its curvature — and the peak is sought in the
    residual. A straight-line baseline is not enough: the curvature at the ends
    of the window then dominates and the search lands on an edge.

    The peak must be an INTERIOR maximum and must stand ``min_snr`` times above
    the point-to-point scatter, so noise wiggles are not reported as structure.
    """
    q = np.asarray(q, float)
    i = np.asarray(i, float)
    m = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    q, i = q[m], i[m]
    order = np.argsort(q)
    q, i = q[order], i[order]
    if q_window is None:
        q_window = default_window(q)
    if q_window:
        lo, hi = q_window
        sel = (q >= lo) & (q <= hi)
        # only honour the window if it leaves enough data to work with
        if sel.sum() >= 10:
            q, i = q[sel], i[sel]
    if q.size < 10:
        return None

    lq, li = np.log10(q), np.log10(i)
    try:
        base = np.polyval(np.polyfit(lq, li, baseline_deg), lq)
    except Exception:  # noqa: BLE001
        return None
    r = li - base

    k = 5
    if r.size < k + 4:
        return None
    smooth = np.convolve(r, np.ones(k) / k, mode="valid")
    qs = q[k // 2: q.size - k // 2]
    lqs = np.log10(qs)
    j = int(np.argmax(smooth))
    if j == 0 or j == smooth.size - 1:
        return None                       # runs into the window edge: not a peak

    noise = float(np.std(r - np.interp(q, qs, smooth)))
    prom = float(smooth[j] - 0.5 * (smooth[:j].min() + smooth[j:].min()))
    if noise <= 0 or prom / noise < min_snr:
        return None

    # parabolic refinement in log Q for sub-point accuracy
    y0, y1, y2 = smooth[j - 1], smooth[j], smooth[j + 1]
    den = y0 - 2 * y1 + y2
    dx = 0.5 * (y0 - y2) / den if den else 0.0
    dx = max(-1.0, min(1.0, dx))
    q_peak = float(10 ** (lqs[j] + dx * (lqs[j + 1] - lqs[j - 1]) / 2))
    return Peak(q=q_peak, d=d_of_q(q_peak), snr=float(prom / noise))


# --------------------------------------------------------------------------- #
# 3) peak position from a fitted model
# --------------------------------------------------------------------------- #
def from_fit(result) -> Peak | None:
    """Repeat distance from a fitted sasmodels result, if its model has a peak
    position (or reports the spacing directly). Propagates the uncertainty."""
    if result is None or not getattr(result, "ok", False):
        return None
    params = getattr(result, "params", None) or {}
    uncs = getattr(result, "uncertainties", None) or {}
    for name in PEAK_D_PARAMS:                       # model reports d directly
        if name in params and params[name]:
            d = float(params[name])
            return Peak(q=(2 * math.pi / d if d else 0.0), d=d, snr=float("nan"),
                        method=f"fitted ({result.model_name}.{name})",
                        d_err=_finite(uncs.get(name)))
    for name in PEAK_Q_PARAMS:                       # model reports Q_peak
        if name in params and params[name]:
            qp = float(params[name])
            if qp <= 0:
                continue
            dq = _finite(uncs.get(name))
            d_err = (2 * math.pi * dq / qp ** 2) if dq else None
            return Peak(q=qp, d=d_of_q(qp), snr=float("nan"),
                        method=f"fitted ({result.model_name}.{name})", d_err=d_err)
    return None


def _finite(x):
    try:
        v = float(x)
        return v if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def prompt_hint(intent: Intent) -> str:
    """Extra instruction for the model-selection agent when the experiment is
    about a repeat distance: make it look for the peak."""
    if not intent.wanted:
        return ""
    win = ""
    if intent.q_window:
        lo, hi = intent.q_window
        win = (f" The proposal points to a peak near Q = {lo:.4g}-{hi:.4g} 1/A "
               f"(d = {2 * math.pi / hi:.0f}-{2 * math.pi / lo:.0f} A); check that "
               "region specifically.")
    return (
        "\nREPEAT-DISTANCE EXPERIMENT: the proposal's goal is a periodic repeat "
        "distance (d = 2*pi/Q_peak), so the SCIENTIFICALLY IMPORTANT feature is the "
        "position of the correlation/Bragg peak, not merely the overall decay. "
        "Include at least one peak-bearing model among your candidates "
        f"({', '.join(PEAK_MODELS[:4])}) so the peak position is actually fitted and "
        "reported with an uncertainty, and do NOT choose a Q-window that cuts the "
        "peak out." + win
    )


# --------------------------------------------------------------------------- #
# 4) per-group reporting
# --------------------------------------------------------------------------- #
def group_peaks(members, intent: Intent, namemap=None) -> list[dict]:
    """Model-free peak + d-spacing for each member of a group."""
    from .loaders import load_iq
    short = namemap.short if namemap is not None else (lambda x: x)
    rows = []
    for ds in members:
        path = ds.merged_path or ds.iq_path
        if not path:
            continue
        try:
            iq = load_iq(path)
        except Exception as e:  # noqa: BLE001
            logger.debug("d-spacing: could not load %s: %s", ds.output_name, e)
            continue
        pk = find_peak(iq.mod_q, iq.intensity, intent.q_window)
        rows.append({
            "name": short(ds.output_name),
            "condition": ds.temperature or "",
            "q_peak": pk.q if pk else None,
            "d": pk.d if pk else None,
            "snr": pk.snr if pk else None,
        })
    return rows


def build_table(group_label: str, group_id: str, rows: list[dict]):
    """TableSpec of Q_peak and d = 2*pi/Q_peak for a group. None if no peak found."""
    from ..models import TableSpec
    if not rows or not any(r.get("d") for r in rows):
        return None
    body = []
    for r in rows:
        if r.get("d"):
            body.append([r["name"], f"{r['q_peak']:.4f}", f"{r['d']:.1f}",
                         f"{r['d'] / 10:.1f}", f"{r['snr']:.1f}"])
        else:
            body.append([r["name"], "—", "—", "—", "—"])
    return TableSpec(
        caption=(f"Repeat distance for {group_label}: the correlation-peak position "
                 r"$Q_\mathrm{peak}$ located model-free (background removed in "
                 r"log-log), and the real-space repeat $d = 2\pi/Q_\mathrm{peak}$. "
                 "SNR is the peak height over the local point-to-point scatter; a "
                 "dash means no peak stood clearly above the noise."),
        label=f"tab:dspacing_{re.sub(r'[^0-9A-Za-z]+', '_', group_id)}",
        headers=["Member", "Q_peak (1/A)", "d (A)", "d (nm)", "SNR"],
        rows=body, fontsize="footnotesize", colspec="l r r r r")
