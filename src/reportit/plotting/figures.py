"""Generate report figures as PNGs. Headless (Agg).

- overlay_iq:  log-log I(Q) overlay of a group's members (optionally comparing
               output variants), styled after eqplot.plotiq.
- plot_2d:     I(Qx,Qy) scatter map with log color scale, equal aspect.
"""

from __future__ import annotations

import logging
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


def _curve_path(ds: Dataset, prefer_merged: bool) -> Path | None:
    if prefer_merged and ds.merged_path and ds.merged_path.is_file():
        return ds.merged_path
    return ds.iq_path


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
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, ds in enumerate(members):
        path = _curve_path(ds, prefer_merged)
        if not path or not Path(path).is_file():
            continue
        try:
            iq = load_iq(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", path, e)
            continue
        q, i, _, _ = clean_low_q(np.asarray(iq.mod_q), np.asarray(iq.intensity))
        mask = (q > 0) & (i > 0) & np.isfinite(q) & np.isfinite(i)
        if mask.sum() < 2:
            continue
        label = ds.output_name
        if compare_variants:
            label = f"{ds.output_name} [{ds.variant}]"
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
