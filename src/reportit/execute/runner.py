"""Carry out an AnalysisStrategy: build figures, metrics tables, and fits per group."""

from __future__ import annotations

import logging
from pathlib import Path

from ..analysis import fit as fitmod
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
    def __init__(self, datasets: list[Dataset], fig_dir: Path):
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
        fit_by_group = {fp.group_id: fp for fp in strategy.fit_plans}

        reports: list[GroupReport] = []
        for g in strategy.groups:
            members = self._members_for(g.members, variants, compare)
            if not members:
                logger.warning("group %s has no resolvable members", g.group_id)
                continue

            gr = GroupReport(group=g)
            # metrics for each member
            for ds in members:
                if ds.iq_path:
                    gr.analyses.append(metricsmod.analyze(ds.output_name, ds.variant, ds.iq_path))

            # fit (on a representative member: prefer one with merged extended-Q)
            fit_result = None
            fp = fit_by_group.get(g.group_id)
            rep_idx = self._representative_index(members)
            if fp and fp.should_fit and fp.model:
                rep = members[rep_idx]
                path = rep.merged_path or rep.iq_path
                if path:
                    try:
                        fit_result = fitmod.run_fit(path, fp.model, fp.q_min, fp.q_max)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("fit failed for %s: %s", g.group_id, e)
                    if fit_result and fit_result.ok and gr.analyses:
                        gr.analyses[rep_idx].fit = fit_result

            # 1D overlay figure
            if g.comparison in ("iq1d", "both"):
                fig_path = self.fig_dir / f"{_safe(g.group_id)}_iq.png"
                made = figures.overlay_iq(
                    g.label, members, fig_path,
                    compare_variants=compare, prefer_merged=True,
                    fit=fit_result, fit_member_index=rep_idx,
                )
                if made:
                    gr.figures.append(FigureRef(
                        path=made, caption=_iq_caption(g, compare),
                        label=f"fig:{_safe(g.group_id)}_iq"))

            # 2D map for representative member
            if g.comparison in ("iqxqy2d", "both"):
                rep = members[rep_idx]
                fig_path = self.fig_dir / f"{_safe(g.group_id)}_2d.png"
                made = figures.plot_2d(rep, fig_path)
                if made:
                    gr.figures.append(FigureRef(
                        path=made,
                        caption=f"2D scattering I($Q_x$,$Q_y$) for {rep.output_name}.",
                        label=f"fig:{_safe(g.group_id)}_2d"))

            gr.table = _metrics_table(g.group_id, gr.analyses, fit_result)
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


def _iq_caption(group, compare: bool) -> str:
    base = (f"Log-log I(Q) for {group.label}. Merged (extended-Q) profiles "
            "combining both detector configurations are shown where available "
            "(legend gives the merged filename); otherwise single-configuration data.")
    if group.ordering_key:
        base += f" Ordered by {group.ordering_key}."
    if compare:
        base += " Solid vs dashed compare the two reduction (mask) variants."
    return base


def _metrics_table(group_id: str, analyses, fit) -> TableSpec | None:
    if not analyses:
        return None
    headers = ["Dataset", "variant", "N", "Q min", "Q max", "low-Q slope", "high-Q slope"]
    rows = []
    for a in analyses:
        rows.append([
            a.output_name, a.variant, str(a.n_points),
            _fmt(a.q_min), _fmt(a.q_max), _fmt(a.low_q_slope, 3), _fmt(a.high_q_slope, 3),
        ])
    caption = f"Per-dataset metrics for {group_id}."
    if fit and fit.ok:
        if fit.kind == "guinier":
            caption += (f" Guinier fit: Rg={_fmt(fit.params.get('Rg'))} $\\mathrm{{\\AA}}$, "
                        f"I0={_fmt(fit.params.get('I0'))} cm$^{{-1}}$ "
                        f"(R²={_fmt(fit.r_squared)}).")
        elif fit.kind == "correlation":
            caption += (f" Ornstein-Zernike fit: correlation length "
                        f"$\\xi$={_fmt(fit.params.get('xi'))} $\\mathrm{{\\AA}}$, "
                        f"I0={_fmt(fit.params.get('I0'))} cm$^{{-1}}$ "
                        f"(R²={_fmt(fit.r_squared)}).")
        else:
            caption += (f" {fit.kind} fit: exponent={_fmt(fit.params.get('exponent'))} "
                        f"(R²={_fmt(fit.r_squared)}).")
    return TableSpec(caption=caption, label=f"tab:{_safe(group_id)}",
                     headers=headers, rows=rows)
