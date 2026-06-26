"""Render a ReportModel into LaTeX source using a jinja2 template.

Custom delimiters avoid clashing with LaTeX braces:
  statements <% %>, expressions << >>, comments <# #>.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import ReportModel, TableSpec
from . import latex_utils as L

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        block_start_string="<%", block_end_string="%>",
        variable_start_string="<<", variable_end_string=">>",
        comment_start_string="<#", comment_end_string="#>",
        trim_blocks=True, lstrip_blocks=True,
        autoescape=select_autoescape(enabled_extensions=[], default=False),
    )
    env.globals["render_table"] = _render_table
    return env


def _render_table(table: TableSpec) -> str:
    if table is None or not table.rows:
        return ""
    ncol = len(table.headers)
    colspec = table.colspec or ("l" * ncol)
    size = "\\" + (table.fontsize or "small")
    head = " & ".join(f"\\textbf{{{L.escape(h)}}}" for h in table.headers)
    body_lines = []
    for row in table.rows:
        cells = [L.escape(c) for c in row]
        cells = (cells + [""] * ncol)[:ncol]
        body_lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_lines)
    cap = L.escape_keep_math(table.caption)

    if table.longtable:
        out = (
            f"{{{size}\n"
            f"\\begin{{longtable}}{{{colspec}}}\n"
            f"\\caption{{{cap}}}\\label{{{table.label}}}\\\\\n"
            f"\\toprule\n{head} \\\\\n\\midrule\n\\endfirsthead\n"
            f"\\toprule\n{head} \\\\\n\\midrule\n\\endhead\n"
            f"\\midrule\\multicolumn{{{ncol}}}{{r}}{{\\textit{{continued on next page}}}}\\\\\n\\endfoot\n"
            f"\\bottomrule\n\\endlastfoot\n"
            f"{body}\n"
            f"\\end{{longtable}}\n}}"
        )
        if table.landscape:
            out = "\\begin{landscape}\n" + out + "\n\\end{landscape}"
        return out

    return (
        "\\begin{table}[H]\n\\centering\n" + size + "\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\caption{{{cap}}}\n"
        f"\\label{{{table.label}}}\n\\end{{table}}"
    )


def render(model: ReportModel, mode: str = "comprehensive") -> str:
    env = _make_env()
    template = env.get_template("report.tex.j2")

    groups = []
    for gr in model.group_reports:
        figs = gr.figures
        if mode == "summary":
            figs = [f for f in gr.figures if f.label.endswith("_iq")][:1]  # 1D only
        groups.append({
            "title": L.escape(gr.group.label),
            "description": L.escape_keep_math(gr.group.description),
            "observations": L.escape_keep_math(gr.observations),
            "figures": [{"path": str(f.path), "caption": L.escape_keep_math(f.caption),
                         "label": f.label, "width": f.width} for f in figs],
            "table": gr.table,
        })

    hyp = [{"hypothesis": L.escape(h.hypothesis), "verdict": L.escape(h.verdict),
            "confidence": L.escape(h.confidence), "evidence": L.escape(h.evidence)}
           for h in model.hypothesis_checks]

    appendix = []
    if mode == "comprehensive":
        for t in model.appendix_tables:
            appendix.append({"section_title": L.escape(t.section_title or "Appendix Table"),
                             "table": t})

    sas_sections = _build_sas_sections(model, mode)
    sas_summary = _build_sas_summary(model)

    return template.render(
        mode=mode,
        title=L.escape(model.title),
        generated_at=L.escape(model.generated_at),
        overview=L.escape_keep_math(model.overview),
        science_goals=[L.escape_keep_math(g) for g in
                       (model.context.proposal.science_goals if model.context.proposal else [])],
        catalog_table=model.catalog_table,
        appendix_tables=appendix,
        group_reports=groups,
        sas_sections=sas_sections,
        sas_summary=sas_summary,
        hypothesis_checks=hyp,
        discussion=L.escape_keep_math(model.discussion),
        caveats=[L.escape_keep_math(c) for c in model.caveats],
    )


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return "—"


def _build_sas_sections(model: ReportModel, mode: str) -> list:
    if mode != "comprehensive" or not model.sas_fits:
        return []
    sections = []
    for o in model.sas_fits:
        attempts = "; ".join(
            f"{L.escape(a.get('model',''))} ("
            f"{'accepted' if a.get('verdict')=='accept' else L.escape(str(a.get('verdict','?')).replace('_',' '))}"
            + (f", $\\chi^2_\\nu$={_fmt(a.get('reduced_chisq'),3)}" if a.get('reduced_chisq') else "")
            + ")"
            for a in o.attempts)
        fig = None
        if o.figure:
            fig = {"path": str(o.figure.path),
                   "caption": L.escape_keep_math(o.figure.caption),
                   "label": o.figure.label, "width": "0.8\\textwidth"}
        trend_fig = None
        if getattr(o, "trend_figure", None):
            trend_fig = {"path": str(o.trend_figure.path),
                         "caption": L.escape_keep_math(o.trend_figure.caption),
                         "label": o.trend_figure.label, "width": "0.7\\textwidth"}
        member_table = _member_fit_table(o)
        # fitted vs fixed parameter breakdown
        fitted_fixed = ""
        if o.best:
            fitted = ", ".join(o.best.params.keys())
            fixed = ", ".join(f"{k}={_fmt(v)}" for k, v in (o.best.fixed or {}).items())
            window = ""
            if o.best.fit_qmin is not None and o.best.q_excluded:
                window = (f" Fitted over Q=[{_fmt(o.best.fit_qmin)}, "
                          f"{_fmt(o.best.fit_qmax)}] (excluding {len(o.best.q_excluded)} "
                          "out-of-range points).")
            fitted_fixed = (f"Fitted parameters: {L.escape(fitted)}. "
                            + (f"Fixed: {L.escape(fixed)}. " if fixed else "")
                            + L.escape(window))
        sections.append({
            "title": L.escape(o.label or o.group_id),
            "status": "Accepted" if o.success else "No satisfactory model found",
            "success": o.success,
            "model": L.escape(o.best.model_name) if o.best else "—",
            "model_description": L.escape_keep_math((o.model_description or "")[:2000]),
            "fitted_fixed": fitted_fixed,
            "chisq": _fmt(o.best.reduced_chisq, 3) if (o.best and o.best.reduced_chisq) else "—",
            "rationale": L.escape_keep_math(o.rationale),
            "critique": L.escape_keep_math(o.critique),
            "attempts": attempts,
            "figure": fig,
            "trend_figure": trend_fig,
            "member_table": member_table,
            "dataset": L.escape(o.dataset_name),
        })
    return sections


def _member_fit_table(o):
    """Comprehensive per-member fit table: EVERY member, ALL fitted parameters
    (value ± error), and reduced chi^2 — one table for the whole group."""
    fits = getattr(o, "member_fits", None) or []
    if len(fits) < 2:
        return None
    # union of fitted parameter names, in a stable order
    pnames: list[str] = []
    for f in fits:
        for p in (f.get("params") or {}):
            if p not in pnames:
                pnames.append(p)
    if not pnames:
        return None
    headers = ["Member", "Condition"] + [f"{p} (±)" for p in pnames] + ["chi2_nu"]
    rows = []
    for f in fits:
        params = f.get("params") or {}
        uncs = f.get("uncertainties") or {}
        cells = [f.get("name", ""), str(f.get("condition", ""))]
        for p in pnames:
            v = params.get(p)
            u = uncs.get(p, 0) or 0
            cells.append(f"{_fmt(v)} ± {_fmt(u)}" if (v is not None and u) else _fmt(v))
        cells.append(_fmt(f.get("reduced_chisq"), 3))
        rows.append(cells)
    # many parameters -> shrink font and go landscape so it fits
    ncol = len(headers)
    fontsize = "scriptsize" if ncol > 6 else "footnotesize"
    landscape = ncol >= 7
    return TableSpec(
        caption=f"Fitted parameters of the {o.best.model_name if o.best else 'chosen'} "
                f"model for every member of {o.label} (value ± uncertainty), with "
                f"reduced chi-squared.",
        label=f"tab:trend_{_safe(o.group_id)}",
        headers=headers, rows=rows,
        longtable=landscape, landscape=landscape, fontsize=fontsize)


def _build_sas_summary(model: ReportModel) -> dict | None:
    if not model.sas_fits:
        return None
    rows = []  # render_table escapes cells — pass raw text
    for o in model.sas_fits:
        rows.append([
            o.label or o.group_id,
            o.best.model_name if o.best else "—",
            _fmt(o.best.reduced_chisq, 3) if (o.best and o.best.reduced_chisq) else "—",
            "yes" if o.success else "no",
        ])
    table = TableSpec(
        caption="Model-based fitting summary: best model and reduced chi-squared per "
                "group, and whether the critic accepted the fit.",
        label="tab:sasfit_summary",
        headers=["Group", "Best model", "Reduced chi2", "Accepted"], rows=rows,
        fontsize="small")
    return {"table": table}


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))


def write_tex(model: ReportModel, out_dir: Path, mode: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = render(model, mode=mode)
    path = out_dir / f"report_{mode}.tex"
    path.write_text(tex)
    return path
