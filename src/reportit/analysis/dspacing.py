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

# Models worth trying when a repeat distance is the goal — every one of these
# reports either a peak position or the spacing itself. NOTE: plain `lamellar`
# and `lamellar_hg` are single-lamella FORM FACTORS with no spacing parameter at
# all, so they cannot answer a repeat-distance question; the lamellar models that
# can are the STACK ones below.
PEAK_MODELS = ("broad_peak", "gaussian_peak", "peak_lorentz",
               "lamellar_stack_caille", "lamellar_hg_stack_caille",
               "lamellar_stack_paracrystal", "teubner_strey")
# forced into the candidate set when the experiment targets a repeat distance
FORCE_MODELS = ("broad_peak", "gaussian_peak", "lamellar_stack_paracrystal")


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
    width_q: float | None = None    # FWHM in Q (1/A)
    q_err: float | None = None
    width_err: float | None = None
    amplitude: float | None = None

    @property
    def width_d(self) -> float | None:
        """Spread in real space implied by the peak width (dd ~ 2*pi*dQ/Q^2)."""
        if not self.width_q or self.q <= 0:
            return None
        return 2 * math.pi * self.width_q / self.q ** 2


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


def find_peaks(q, i, q_window: tuple[float, float] | None = None, *,
               min_snr: float = 2.5, baseline_deg: int = 4,
               min_sep_dex: float = 0.06, max_peaks: int = 6) -> list[Peak]:
    """ALL correlation peaks in a curve, strongest first.

    A stacked/lamellar system does not give one peak: there is a dominant order
    plus weaker ones, and describing them with a single broad component (as a
    lone `broad_peak` fit does) smears real structure into one wide feature. So
    every local maximum of the background-subtracted residual that clears
    ``min_snr`` is reported, subject to a minimum separation of ``min_sep_dex``
    decades in Q so one peak is not counted twice.

    Widths here are rough (from the half-maximum crossings of the residual);
    :func:`fit_peak_model` refines position AND width properly by least squares.
    """
    prep = _residual(q, i, q_window, baseline_deg)
    if prep is None:
        return []
    qs, smooth, noise = prep
    if noise <= 0:
        return []
    lqs = np.log10(qs)
    found: list[Peak] = []
    for j in range(1, smooth.size - 1):
        if not (smooth[j] > smooth[j - 1] and smooth[j] >= smooth[j + 1]):
            continue
        prom = float(smooth[j] - 0.5 * (smooth[:j].min() + smooth[j:].min()))
        snr = prom / noise
        if snr < min_snr:
            continue
        # parabolic refinement in log Q
        y0, y1, y2 = smooth[j - 1], smooth[j], smooth[j + 1]
        den = y0 - 2 * y1 + y2
        dx = 0.5 * (y0 - y2) / den if den else 0.0
        dx = max(-1.0, min(1.0, dx))
        qp = float(10 ** (lqs[j] + dx * (lqs[j + 1] - lqs[j - 1]) / 2))
        found.append(Peak(q=qp, d=d_of_q(qp), snr=float(snr),
                          width_q=_half_width(qs, smooth, j)))
    found.sort(key=lambda p: p.snr, reverse=True)
    kept: list[Peak] = []
    for p in found:
        if any(abs(np.log10(p.q) - np.log10(k.q)) < min_sep_dex for k in kept):
            continue
        kept.append(p)
        if len(kept) >= max_peaks:
            break
    kept.sort(key=lambda p: p.q)
    return kept


def _half_width(qs, smooth, j) -> float | None:
    """Rough FWHM in Q from the half-maximum crossings around index ``j``."""
    half = smooth[j] - 0.5 * (smooth[j] - min(smooth.min(), 0.0))
    lo = hi = None
    for a in range(j, 0, -1):
        if smooth[a] <= half:
            lo = qs[a]
            break
    for b in range(j, smooth.size):
        if smooth[b] <= half:
            hi = qs[b]
            break
    if lo is None or hi is None or hi <= lo:
        return None
    return float(hi - lo)


def _residual(q, i, q_window, baseline_deg):
    """(qs, smoothed residual, noise) after removing a smooth log-log baseline."""
    q = np.asarray(q, float)
    i = np.asarray(i, float)
    m = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    q, i = q[m], i[m]
    o = np.argsort(q)
    q, i = q[o], i[o]
    if q_window is None:
        q_window = default_window(q)
    lo, hi = q_window
    sel = (q >= lo) & (q <= hi)
    if sel.sum() >= 10:
        q, i = q[sel], i[sel]
    if q.size < 10:
        return None
    lq, li = np.log10(q), np.log10(i)
    try:
        r = li - np.polyval(np.polyfit(lq, li, baseline_deg), lq)
    except Exception:  # noqa: BLE001
        return None
    k = 5
    if r.size < k + 4:
        return None
    smooth = np.convolve(r, np.ones(k) / k, mode="valid")
    qs = q[k // 2: q.size - k // 2]
    noise = float(np.std(r - np.interp(q, qs, smooth)))
    return qs, smooth, noise


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
        "For a LAMELLAR/stacked system use a model that actually reports a "
        "spacing: lamellar_stack_caille, lamellar_hg_stack_caille or "
        "lamellar_stack_paracrystal all have a d_spacing parameter, whereas "
        "plain `lamellar` and `lamellar_hg` are single-lamella form factors "
        "with NO spacing parameter and cannot answer the question. "
        "Include at least one peak-bearing model among your candidates "
        f"({', '.join(PEAK_MODELS[:5])}), and do NOT choose a Q-window that cuts "
        "the peak out.\n"
        "COUNT THE PEAKS FIRST. A stacked/lamellar system commonly shows SEVERAL "
        "finer peaks — one dominant plus weaker ones. A single broad_peak component "
        "will happily fit ONE wide bump straight across all of them: that is a "
        "misfit, and the position and width it reports belong to no actual peak. If "
        "the curve has more than one peak, say so and prefer a description that "
        "gives EACH peak its own position and width (an empirical correlation-type "
        "background plus one Gaussian per peak has already been fitted for this "
        "curve and is reported separately). Where the peak positions and widths are "
        "already measured that way, a further single-model sasmodels fit adds "
        "little — judge candidates on whether they reproduce the peak STRUCTURE, "
        "and say plainly if none does." + win
    )


# --------------------------------------------------------------------------- #
# 4) per-group reporting
# --------------------------------------------------------------------------- #
def group_peaks(members, intent: Intent, namemap=None) -> list[dict]:
    """Every resolved peak of every member: position, width, and d = 2*pi/Q.

    Uses the empirical background + Gaussian-peaks fit, so a stacked system with
    several finer peaks is described peak by peak rather than smeared into one
    broad component.
    """
    from .loaders import load_iq
    short = namemap.short if namemap is not None else (lambda x: x)
    rows: list[dict] = []
    fits: list = []
    for ds in members:
        path = ds.merged_path or ds.iq_path
        if not path:
            continue
        try:
            iq = load_iq(path)
        except Exception as e:  # noqa: BLE001
            logger.debug("d-spacing: could not load %s: %s", ds.output_name, e)
            continue
        fit = fit_peak_model(iq.mod_q, iq.intensity, getattr(iq, "error", None),
                             q_window=intent.q_window)
        fits.append((short(ds.output_name), fit))
        if not fit.ok or not fit.peaks:
            rows.append({"name": short(ds.output_name), "order": None,
                         "q_peak": None, "d": None, "width_q": None,
                         "log_rms": fit.log_rms})
            continue
        for k, pk in enumerate(fit.peaks, 1):
            rows.append({
                "name": short(ds.output_name),
                "order": k, "n_peaks": fit.n_peaks,
                "q_peak": pk.q, "q_err": pk.q_err,
                "d": pk.d, "d_err": pk.d_err,
                "width_q": pk.width_q, "width_err": pk.width_err,
                "log_rms": fit.log_rms,
            })
    return rows, fits


def build_table(group_label: str, group_id: str, rows: list[dict]):
    """TableSpec of Q_peak and d = 2*pi/Q_peak for a group. None if no peak found."""
    from ..models import TableSpec
    if not rows or not any(r.get("d") for r in rows):
        return None
    def _pm(v, e, nd=4):
        if v is None:
            return "—"
        return f"{v:.{nd}f} ± {e:.{nd}f}" if e else f"{v:.{nd}f}"

    body = []
    for r in rows:
        if not r.get("d"):
            body.append([r["name"], "—", "—", "—", "—", "—"])
            continue
        body.append([
            r["name"],
            str(r.get("order") or 1),
            _pm(r.get("q_peak"), r.get("q_err")),
            _pm(r.get("d"), r.get("d_err"), 1),
            f"{r['d'] / 10:.1f}",
            _pm(r.get("width_q"), r.get("width_err")),
        ])
    return TableSpec(
        caption=(f"Peaks resolved for {group_label}. The curve is described "
                 "empirically as a correlation-type background plus one Gaussian "
                 "per peak, so each peak gets its OWN position and width rather "
                 "than being absorbed into a single broad component. "
                 r"$d = 2\pi/Q_\mathrm{peak}$ is the real-space repeat; the width "
                 "is the fitted FWHM in Q (a broader peak means a less ordered, "
                 "more widely distributed spacing). Values are given with their "
                 "1-sigma fit uncertainties; a dash means no peak was resolved "
                 "well enough to quote."),
        label=f"tab:dspacing_{re.sub(r'[^0-9A-Za-z]+', '_', group_id)}",
        headers=["Member", "Peak", "Q_peak (1/A)", "d (A)", "d (nm)", "FWHM (1/A)"],
        rows=body, fontsize="footnotesize", colspec="l c r r r r")


# --------------------------------------------------------------------------- #
# 5) empirical multi-peak fit: correlation background + N Gaussian peaks
# --------------------------------------------------------------------------- #
@dataclass
class MultiPeakFit:
    """An empirical description of a curve as background + N Gaussian peaks."""

    n_peaks: int = 0
    peaks: list = field(default_factory=list)     # list[Peak], low-Q first
    # RMS of log10(data) - log10(model): the vertical gap a human sees. A real
    # reduced chi-squared is not quoted because the fit is done in log space with
    # normalised weights, where it would not mean what the name implies.
    log_rms: float | None = None
    bic: float | None = None
    q: object = None                              # arrays for plotting
    i_data: object = None
    i_model: object = None
    background: object = None
    ok: bool = False
    note: str = ""


def _bg_model(q, logA, n, logC, xi, m, logB):
    """correlation_length-style background: Porod + Lorentzian + flat."""
    A, C, B = 10.0 ** logA, 10.0 ** logC, 10.0 ** logB
    return A * q ** (-n) + C / (1.0 + (q * xi) ** m) + B


def _full_model(q, p, n_peaks):
    out = _bg_model(q, *p[:6])
    for k in range(n_peaks):
        amp, q0, logs = p[6 + 3 * k: 9 + 3 * k]
        sig = 10.0 ** logs
        out = out + abs(amp) * np.exp(-0.5 * ((q - q0) / sig) ** 2)
    return out


def fit_peak_model(q, i, err=None, *, q_window=None, max_peaks: int = 4,
                   seed_peaks: list | None = None) -> MultiPeakFit:
    """Fit background + N Gaussian peaks and pick N empirically.

    Rationale (operator feedback): a stacked system shows SEVERAL finer peaks. A
    single `broad_peak` component fits one wide bump across all of them, which
    smears real structure and reports a position and width that belong to no
    actual peak. Describing the curve as a correlation-type background plus a
    Gaussian per peak is empirical but honest: every peak gets its own position
    and width, each with an uncertainty.

    N is chosen by BIC, so an extra peak has to earn its parameters rather than
    simply lowering chi-squared.
    """
    from scipy.optimize import least_squares

    q = np.asarray(q, float)
    i = np.asarray(i, float)
    e = np.asarray(err, float) if err is not None else None
    m = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    if e is not None:
        m &= np.isfinite(e)
    q, i = q[m], i[m]
    e = e[m] if e is not None else None
    o = np.argsort(q)
    q, i = q[o], i[o]
    e = e[o] if e is not None else None
    # Fit over the SAME window the peaks are sought in. Fitting the full range
    # while seeding from a window makes the background work hard on the noisy
    # tail, and the peak terms then look unnecessary to the model-selection
    # criterion — peaks that are plainly visible get dropped.
    if q_window is None:
        q_window = default_window(q)
    lo, hi = q_window
    sel = (q >= lo) & (q <= hi)
    if sel.sum() >= 15:
        q, i, e = q[sel], i[sel], (e[sel] if e is not None else None)
    if q.size < 15:
        return MultiPeakFit(note="too few points")

    li = np.log10(i)
    # weight by the relative error, in log space; floor keeps a few good points
    # from dominating the whole fit
    if e is not None and np.all(e >= 0):
        rel = np.clip(e / np.maximum(i, 1e-30), 0.005, 1.0)
        w = 1.0 / (rel / np.log(10))
    else:
        w = np.ones_like(q)
    w = w / np.median(w)

    seeds = list(seed_peaks or find_peaks(q, i, q_window))
    cands: list[MultiPeakFit] = []

    for n_peaks in range(0, min(max_peaks, len(seeds)) + 1):
        p0, lo_b, hi_b = _seed_params(q, i, seeds[:n_peaks])
        try:
            res = least_squares(
                lambda p: (np.log10(np.maximum(_full_model(q, p, n_peaks), 1e-30)) - li) * w,
                p0, bounds=(lo_b, hi_b), max_nfev=4000)
        except Exception as ex:  # noqa: BLE001
            logger.debug("multi-peak fit n=%d failed: %s", n_peaks, ex)
            continue
        npar = len(p0)
        chi2 = float(np.sum(res.fun ** 2))
        bic = q.size * np.log(chi2 / q.size) + npar * np.log(q.size)
        model = _full_model(q, res.x, n_peaks)
        peaks = _keep_real_peaks(_peaks_from_params(res, n_peaks), q)
        log_rms = float(np.sqrt(np.mean(
            (np.log10(np.maximum(model, 1e-30)) - li) ** 2)))
        note = ""
        if len(peaks) < n_peaks:
            note = (f"{n_peaks - len(peaks)} fitted component(s) rejected as "
                    "background rather than resolved peaks")
        cands.append(MultiPeakFit(
            n_peaks=len(peaks), log_rms=log_rms, bic=float(bic),
            q=q, i_data=i, i_model=model,
            background=_bg_model(q, *res.x[:6]), ok=True, peaks=peaks, note=note))

    if not cands:
        return MultiPeakFit(note="all fits failed")
    # Prefer a fit whose every component survived as a real peak; among those the
    # best BIC wins. Only if none is clean do we fall back to the best BIC overall
    # and report the components that did survive.
    clean = [c for c in cands if not c.note]
    best = min(clean or cands, key=lambda c: c.bic if c.bic is not None else np.inf)
    best.peaks.sort(key=lambda p: p.q)
    return best


def _seed_params(q, i, seeds):
    """Initial values and bounds for the composite model."""
    qmin, qmax = float(q.min()), float(q.max())
    imax = float(np.max(i))
    imin = max(float(np.min(i)), 1e-12)
    p0 = [np.log10(max(imax * qmin ** 3, 1e-12)), 3.0,
          np.log10(max(imax * 0.1, 1e-12)), 1.0 / max(qmin, 1e-6) / 10, 2.0,
          np.log10(imin)]
    lo = [-30.0, 0.0, -30.0, 1.0, 0.5, -30.0]
    hi = [30.0, 6.0, 30.0, 1e5, 8.0, np.log10(max(imax, 1e-9))]
    for pk in seeds:
        amp = max(imax * 0.05, 1e-12)
        sig = (pk.width_q / 2.355) if pk.width_q else (pk.q * 0.1)
        sig = max(sig, (qmax - qmin) / 200.0)
        p0 += [amp, float(pk.q), float(np.log10(sig))]
        lo += [0.0, max(qmin, pk.q * 0.7), np.log10((qmax - qmin) / 400.0)]
        hi += [imax * 50, min(qmax, pk.q * 1.4), np.log10((qmax - qmin) / 3.0)]
    return np.array(p0, float), np.array(lo, float), np.array(hi, float)


def _peaks_from_params(res, n_peaks) -> list:
    """Peaks with 1-sigma errors from the least-squares Jacobian."""
    perr = _param_errors(res)
    out = []
    for k in range(n_peaks):
        amp, q0, logs = res.x[6 + 3 * k: 9 + 3 * k]
        sig = 10.0 ** logs
        fwhm = 2.3548 * sig
        q0e = perr[7 + 3 * k] if perr is not None else None
        se = (perr[8 + 3 * k] if perr is not None else None)
        fwhm_e = (2.3548 * sig * np.log(10) * se) if se else None
        d = d_of_q(q0) if q0 > 0 else float("nan")
        d_err = (2 * math.pi * q0e / q0 ** 2) if (q0e and q0 > 0) else None
        out.append(Peak(q=float(q0), d=float(d), snr=float("nan"),
                        method="fitted (background + Gaussian peaks)",
                        d_err=d_err, width_q=float(fwhm),
                        q_err=float(q0e) if q0e else None,
                        width_err=float(fwhm_e) if fwhm_e else None,
                        amplitude=float(abs(amp))))
    return out


def _keep_real_peaks(peaks: list, q) -> list:
    """Discard fitted components that are not peaks.

    Least squares will happily add a very broad, poorly-located Gaussian that
    simply absorbs background curvature, or a second component sitting on top of
    the first. Such a component is not a measurement, so it is rejected:

      * width greater than a third of the fitted Q-range -> that is background;
      * centre uncertainty worse than its own FWHM, or worse than 20% of the
        peak position itself -> not located well enough to be a measurement;
      * centre within one FWHM of a stronger peak already kept -> duplicate.
    """
    span = float(np.max(q) - np.min(q))
    keep: list = []
    for p in sorted(peaks, key=lambda x: -(x.amplitude or 0.0)):
        if not p.width_q or p.width_q <= 0:
            continue
        if p.width_q > span / 3.0:
            continue
        if p.q_err and p.q_err > p.width_q:
            continue
        if p.q_err and p.q > 0 and p.q_err / p.q > 0.20:
            continue
        if any(abs(p.q - k.q) < max(p.width_q, k.width_q or 0) for k in keep):
            continue
        keep.append(p)
    keep.sort(key=lambda x: x.q)
    return keep


def _param_errors(res):
    try:
        _, sv, vt = np.linalg.svd(res.jac, full_matrices=False)
        keep = sv > 1e-12 * sv[0]
        cov = (vt[keep].T / sv[keep] ** 2) @ vt[keep]
        dof = max(1, res.fun.size - res.x.size)
        cov = cov * (float(np.sum(res.fun ** 2)) / dof)
        return np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:  # noqa: BLE001
        return None
