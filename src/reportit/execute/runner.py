"""Carry out an AnalysisStrategy: build figures, metrics tables, and fits per group."""

from __future__ import annotations

import logging
from pathlib import Path

from ..analysis import metrics as metricsmod
from ..models import (
    AnalysisStrategy,
    Dataset,
    FigureRef,
    FitPlan,
    GroupReport,
    TableSpec,
)
from ..plotting import figures

logger = logging.getLogger(__name__)


def _fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


class Runner:
    def __init__(self, datasets: list[Dataset], fig_dir: Path, namemap=None,
                 d_intent=None):
        self.namemap = namemap
        self.d_intent = d_intent
        # (label, MultiPeakFit) across every group, for one summary grid
        self.peak_fits: list = []
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        # index output_name -> {variant: Dataset}
        self.index: dict[str, dict[str, Dataset]] = {}
        for d in datasets:
            self.index.setdefault(d.output_name, {})[d.variant] = d

    def _members_for(self, names: list[str], variants: list[str], compare: bool) -> list[Dataset]:
        out: list[Dataset] = []
        for name in names:
            by_variant = self.index.get(name, {})
            if not by_variant:
                continue
            if compare:
                for v in variants:
                    if v in by_variant:
                        out.append(by_variant[v])
            else:
                # pick the first requested variant available
                for v in variants:
                    if v in by_variant:
                        out.append(by_variant[v])
                        break
                else:
                    out.append(next(iter(by_variant.values())))
        return out

    def run(self, strategy: AnalysisStrategy) -> list[GroupReport]:
        variants = strategy.variant_decision.variants_used or ["output"]
        compare = strategy.variant_decision.compare and len(variants) > 1
        # ALWAYS prefer merged (extended-Q) profiles for plotting and fitting;
        # overlay_iq falls back to the single-configuration curve per-dataset only
        # when no merged/combined file exists for that sample+condition.
        prefer_merged = True

        reports: list[GroupReport] = []
        for g in strategy.groups:
            members = self._members_for(g.members, variants, compare)
            if not members:
                logger.warning("group %s has no resolvable members", g.group_id)
                continue

            glabel = (self.namemap.shorten_label(g.label)
                      if self.namemap is not None else g.label)
            gr = GroupReport(group=g)
            # descriptive metrics only — NO model fitting in this section. Section 2
            # is purely the data + qualitative observations; all model fitting lives
            # in the Model-Based Fitting section.
            for ds in members:
                if ds.iq_path:
                    gr.analyses.append(metricsmod.analyze(ds.output_name, ds.variant, ds.iq_path))

            rep_idx = self._representative_index(members)

            # 1D overlay figure (data only, merged extended-Q, no fit curve)
            if g.comparison in ("iq1d", "both"):
                fig_path = self.fig_dir / f"{_safe(g.group_id)}_iq.png"
                made = figures.overlay_iq(
                    glabel, members, fig_path,
                    compare_variants=compare, prefer_merged=prefer_merged,
                    fit=None, namemap=self.namemap,
                )
                if made:
                    gr.figures.append(FigureRef(
                        path=made, caption=_iq_caption(g, compare, glabel),
                        label=f"fig:{_safe(g.group_id)}_iq"))

            # 2D map for representative member
            if g.comparison in ("iqxqy2d", "both"):
                rep = members[rep_idx]
                fig_path = self.fig_dir / f"{_safe(g.group_id)}_2d.png"
                made = figures.plot_2d(rep, fig_path)
                if made:
                    gr.figures.append(FigureRef(
                        path=made,
                        caption=f"2D scattering I($Q_x$,$Q_y$) for "
                                f"{(self.namemap.short(rep.output_name) if self.namemap else rep.output_name)}.",
                        label=f"fig:{_safe(g.group_id)}_2d"))

            gr.table = _metrics_table(g.group_id, gr.analyses, None,
                                      namemap=self.namemap, caption_label=glabel)

            # repeat-distance experiment: locate the correlation peak on every
            # member and report d = 2*pi/Q_peak alongside the curves
            if self.d_intent is not None and self.d_intent.wanted:
                from ..analysis import dspacing
                rows, pfits = dspacing.group_peaks(members, self.d_intent,
                                                   self.namemap)
                dtab = dspacing.build_table(glabel, g.group_id, rows)
                if dtab is not None:
                    gr.extra_tables.append(dtab)
                # Collected for a single summary grid rather than a
                # full-width figure each: one curve does not need the page
                # width, and a grid lets the series be compared at a glance.
                self.peak_fits.extend(
                    (lbl, pf) for lbl, pf in pfits if pf.ok and pf.peaks)
            reports.append(gr)
        return reports

    @staticmethod
    def _representative_index(members: list[Dataset]) -> int:
        for i, m in enumerate(members):
            if m.merged_path:
                return i
        return 0


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def _iq_caption(group, compare: bool, label: str | None = None) -> str:
    base = (f"Log-log I(Q) for {label or group.label}. Merged (extended-Q) profiles "
            "combining both detector configurations are shown where available; "
            "otherwise single-configuration data.")
    if group.ordering_key:
        base += f" Ordered by {group.ordering_key}."
    if compare:
        base += " Solid vs dashed compare the two reduction (mask) variants."
    return base


def _metrics_table(group_id: str, analyses, fit, namemap=None,
                   caption_label: str | None = None) -> TableSpec | None:
    if not analyses:
        return None
    # Descriptive only — Q-range and point count. NO slopes here: a log-log slope
    # depends strongly on the (unstated) Q-range used, so quoting it is misleading.
    # Quantitative analysis belongs to the Model-Based Fitting section.
    headers = ["Dataset", "variant", "N points", "Q min (1/A)", "Q max (1/A)"]
    short = namemap.short if namemap is not None else (lambda x: x)
    rows = []
    for a in analyses:
        rows.append([
            short(a.output_name), a.variant, str(a.n_points),
            _fmt(a.q_min), _fmt(a.q_max),
        ])
    return TableSpec(caption="Measured Q-range and point count for "
                             f"{caption_label or group_id}.",
                     label=f"tab:{_safe(group_id)}", headers=headers, rows=rows)
