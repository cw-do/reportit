"""Generate report figures as PNGs. Headless (Agg).

- overlay_iq:  log-log I(Q) overlay of a group's members (optionally comparing
               output variants), styled after eqplot.plotiq.
- plot_2d:     I(Qx,Qy) scatter map with log color scale, equal aspect.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from ..analysis.clean import clean_low_q  # noqa: E402
from ..analysis.loaders import load_iq, load_iqxqy  # noqa: E402
from ..models import Dataset, FitResult  # noqa: E402

logger = logging.getLogger(__name__)

_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "x", "+"]


def _plot_model_line(ax, result, color, lw=1.7, label=None):
    """Plot the fitted model: solid inside the fit window, dashed where it extends
    beyond it (so the reader sees how the fit deviates outside the fitted range).
    Falls back to the fitted-window model if no full-range curve is available."""
    qf = np.asarray(getattr(result, "q_full", []), float)
    mf = np.asarray(getattr(result, "i_model_full", []), float)
    if qf.size == 0 or mf.size != qf.size:
        qf = np.asarray(result.q, float)
        mf = np.asarray(result.i_model, float)
    pos = (qf > 0) & (mf > 0) & np.isfinite(qf) & np.isfinite(mf)
    if pos.sum() < 2:
        return
    # full extent, dashed
    ax.plot(qf[pos], mf[pos], "--", color=color, lw=lw * 0.85, alpha=0.75)
    # in-window, solid (drawn on top of the dashed)
    win = pos.copy()
    if result.fit_qmin is not None:
        win &= qf >= result.fit_qmin
    if result.fit_qmax is not None:
        win &= qf <= result.fit_qmax
    ax.plot(qf[win], mf[win], "-", color=color, lw=lw, label=label)


def _curve_path(ds: Dataset, prefer_merged: bool) -> Path | None:
    if prefer_merged and ds.merged_path and ds.merged_path.is_file():
        return ds.merged_path
    return ds.iq_path


def _legend_label(merged_path) -> str:
    """Legend label for a merged curve = its filename core (e.g. 15_30C_4m10a_2.5m2.5a)."""
    stem = Path(merged_path).name
    stem = re.sub(r"^merged_", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_Iq\.\w+$", "", stem, flags=re.IGNORECASE)
    return stem


def overlay_iq(
    group_label: str,
    members: list[Dataset],
    out_path: Path,
    *,
    compare_variants: bool = False,
    prefer_merged: bool = True,
    fit: FitResult | None = None,
    fit_member_index: int = 0,
) -> Path | None:
    """Log-log overlay of I(Q) for a group's members. Returns out_path or None."""
    plotted = 0
    seen_paths: set[str] = set()
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, ds in enumerate(members):
        path = _curve_path(ds, prefer_merged)
        if not path or not Path(path).is_file():
            continue
        rp = str(Path(path).resolve())
        if rp in seen_paths:  # same merged file shared by two configs — plot once
            continue
        seen_paths.add(rp)
        try:
            iq = load_iq(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", path, e)
            continue
        q, i, _, _ = clean_low_q(np.asarray(iq.mod_q), np.asarray(iq.intensity))
        mask = (q > 0) & (i > 0) & np.isfinite(q) & np.isfinite(i)
        if mask.sum() < 2:
            continue
        is_merged = bool(ds.merged_path) and Path(path) == Path(ds.merged_path)
        label = _legend_label(path) if is_merged else ds.output_name
        if compare_variants:
            label = f"{label} [{ds.variant}]"
        ls = "-" if ds.variant.endswith("mask4") or not compare_variants else "--"
        ax.plot(q[mask], i[mask], marker=_MARKERS[idx % len(_MARKERS)],
                markersize=3, linewidth=1, linestyle=ls, fillstyle="none", label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    # overlay fit curve if provided and physical
    if fit and fit.ok and 0 <= fit_member_index < len(members):
        _overlay_fit(ax, members[fit_member_index], fit, prefer_merged)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"I(Q) (cm$^{-1}$)")
    ax.set_title(group_label)
    if plotted <= 12:
        ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, which="major", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _overlay_fit(ax, ds: Dataset, fit: FitResult, prefer_merged: bool) -> None:
    path = _curve_path(ds, prefer_merged)
    try:
        iq = load_iq(path)
    except Exception:
        return
    q = np.asarray(iq.mod_q)
    q = q[(q > 0) & np.isfinite(q)]
    if q.size == 0:
        return
    qmin, qmax = fit.q_range
    qq = np.linspace(max(qmin, q.min()), min(qmax, q.max()) or q.max(), 100)
    if fit.kind == "guinier" and "Rg" in fit.params:
        rg, i0 = fit.params["Rg"], fit.params["I0"]
        model = i0 * np.exp(-(rg ** 2) * qq ** 2 / 3.0)
        ax.plot(qq, model, "k--", linewidth=1.5, label=f"Guinier fit (Rg={rg:.1f} Å)")
    elif fit.kind == "correlation" and "xi" in fit.params:
        i0 = fit.params["I0"]
        xi = fit.params["xi"]
        bkg = fit.params.get("bkg", 0.0)
        model = i0 / (1.0 + (qq * xi) ** 2) + bkg
        ax.plot(qq, model, "k--", linewidth=1.5, label=f"OZ fit ($\\xi$={xi:.1f} Å)")
    elif fit.kind in ("porod", "powerlaw") and "exponent" in fit.params:
        p, a = fit.params["exponent"], fit.params["prefactor"]
        ax.plot(qq, a * qq ** p, "k--", linewidth=1.5, label=f"power law (p={p:.2f})")


def plot_fit(result, out_path: Path, *, title: str = "") -> Path | None:
    """Plot fitted sasmodels curve over the data (log-log) with a residual panel."""
    q = np.asarray(result.q, float)
    yd = np.asarray(result.i_data, float)
    ym = np.asarray(result.i_model, float)
    if q.size < 3 or ym.size != q.size:
        return None
    m = (q > 0) & (yd > 0) & np.isfinite(yd) & np.isfinite(ym)
    if m.sum() < 3:
        return None

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    # excluded (out-of-window) data, shown faintly for context
    qe = np.asarray(getattr(result, "q_excluded", []), float)
    ye = np.asarray(getattr(result, "i_excluded", []), float)
    if qe.size and ye.size:
        me = (qe > 0) & (ye > 0) & np.isfinite(qe) & np.isfinite(ye)
        ax.plot(qe[me], ye[me], "x", ms=4, color="lightgray",
                label="excluded (out of fit range)")
    ax.errorbar(q[m], yd[m], fmt="o", ms=3, fillstyle="none",
                color="tab:blue", label="data (fitted)")
    _plot_model_line(ax, result, "tab:red", lw=1.8,
                     label=f"{result.model_name} fit (dashed = extrapolated)")
    if getattr(result, "fit_qmin", None) and qe.size:
        ax.axvline(result.fit_qmin, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"I(Q) (cm$^{-1}$)")
    ax.legend(fontsize=8)
    ttl = title or result.model_name
    if result.reduced_chisq is not None:
        ttl += f"   ($\\chi^2_\\nu$={result.reduced_chisq:.1f})"
    ax.set_title(ttl)
    ax.grid(True, which="major", alpha=0.2)

    resid = (yd[m] - ym[m]) / np.where(yd[m] != 0, yd[m], 1)
    axr.axhline(0, color="k", lw=0.8)
    axr.plot(q[m], resid, "o", ms=3, color="tab:gray")
    axr.set_xscale("log")
    axr.set_ylabel("(data-fit)/data")
    axr.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
    axr.grid(True, which="major", alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_group_fits(group_label: str, items: list, out_path: Path) -> Path | None:
    """Overlay every member's fit in one log-log plot: data as markers, fitted
    model as a line (one color per member). items = [(condition_label, SasFitResult)].
    """
    items = [(lbl, r) for lbl, r in items
             if r is not None and r.ok and len(r.q) and len(r.i_model) == len(r.q)]
    if len(items) < 1:
        return None
    cmap = plt.get_cmap("viridis")
    n = len(items)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, (lbl, r) in enumerate(items):
        color = cmap(k / max(n - 1, 1))
        q = np.asarray(r.q, float)
        yd = np.asarray(r.i_data, float)
        ym = np.asarray(r.i_model, float)
        m = (q > 0) & (yd > 0) & np.isfinite(q) & np.isfinite(yd)
        ax.plot(q[m], yd[m], "o", ms=3, fillstyle="none", color=color, label=str(lbl))
        _plot_model_line(ax, r, color, lw=1.5)  # solid in window, dashed outside
        # faint excluded (out-of-window) points for context
        qe = np.asarray(getattr(r, "q_excluded", []), float)
        ye = np.asarray(getattr(r, "i_excluded", []), float)
        if qe.size and ye.size:
            me = (qe > 0) & (ye > 0) & np.isfinite(qe) & np.isfinite(ye)
            ax.plot(qe[me], ye[me], "x", ms=3, color=color, alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"I(Q) (cm$^{-1}$)")
    ax.set_title(group_label)
    if n <= 14:
        ax.legend(fontsize=7, framealpha=0.7, title="markers=data, lines=fit")
    ax.grid(True, which="major", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_trend(label: str, param: str, points: list, out_path: Path,
               *, xlabel: str = "condition", numeric_x: bool = True) -> Path | None:
    """Plot a fitted parameter vs condition (e.g. Rg vs temperature).

    points: list of (x_value, y_value, y_err, x_label_text).
    """
    pts = [p for p in points if p[1] is not None and np.isfinite(p[1])]
    if len(pts) < 2:
        return None
    if numeric_x and all(p[0] is not None for p in pts):
        pts = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ticks = None
    else:
        xs = list(range(len(pts)))
        ticks = [p[3] for p in pts]
    ys = [p[1] for p in pts]
    es = [p[2] if (p[2] and np.isfinite(p[2])) else 0 for p in pts]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3, color="tab:blue")
    if ticks is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(param)
    ax.set_title(f"{label}: {param} vs {xlabel}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_2d(ds: Dataset, out_path: Path, *, markersize: int = 10) -> Path | None:
    if not ds.iqxqy_path or not Path(ds.iqxqy_path).is_file():
        return None
    try:
        d = load_iqxqy(ds.iqxqy_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("2D load failed %s: %s", ds.iqxqy_path, e)
        return None
    qx, qy, inten = np.asarray(d.qx), np.asarray(d.qy), np.asarray(d.intensity)
    mask = np.isfinite(inten) & (inten > 0)
    if mask.sum() < 10:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(qx[mask], qy[mask], c=inten[mask], s=markersize,
                    norm=LogNorm(), cmap="viridis")
    lim_x = float(np.max(np.abs(qx[mask])))
    lim_y = float(np.max(np.abs(qy[mask])))
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$Q_x$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$Q_y$ ($\mathrm{\AA}^{-1}$)")
    ax.set_title(f"{ds.output_name}  I($Q_x$,$Q_y$)")
    fig.colorbar(sc, ax=ax, label=r"I (cm$^{-1}$)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
