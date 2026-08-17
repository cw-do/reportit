"""Render a ReportModel into LaTeX source using a jinja2 template.

Custom delimiters avoid clashing with LaTeX braces:
  statements <% %>, expressions << >>, comments <# #>.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
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

    nm = model.namemap
    slabel = nm.shorten_label if nm is not None else (lambda x: x)
    groups = []
    for gr in model.group_reports:
        figs = gr.figures
        if mode == "summary":
            figs = [f for f in gr.figures if f.label.endswith("_iq")][:1]  # 1D only
        groups.append({
            "title": L.escape(slabel(gr.group.label)),
            "description": L.escape_keep_math(gr.group.description),
            "observations": L.escape_keep_math(gr.observations),
            "figures": [{"path": str(f.path), "caption": L.escape_keep_math(f.caption),
                         "label": f.label, "width": f.width} for f in figs],
            "table": gr.table,
            "extra_tables": list(getattr(gr, "extra_tables", []) or []),
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
    sas_appendix = _build_sas_appendix(model, mode)

    return template.render(
        mode=mode,
        title=L.escape(model.title),
        generated_at=L.escape(model.generated_at),
        reportit_version=L.escape(model.reportit_version or __version__),
        llm_model=L.escape(model.model_name),
        overview=L.escape_keep_math(model.overview),
        science_goals=[L.escape_keep_math(g) for g in
                       (model.context.proposal.science_goals if model.context.proposal else [])],
        catalog_table=model.catalog_table,
        appendix_tables=appendix,
        group_reports=groups,
        summary_figures=[{"path": str(f.path),
                          "caption": L.escape_keep_math(f.caption),
                          "label": f.label, "width": f.width}
                         for f in (model.summary_figures or [])],
        sas_sections=sas_sections,
        sas_summary=sas_summary,
        sas_appendix=sas_appendix,
        hypothesis_checks=hyp,
        discussion=L.escape_keep_math(model.discussion),
        caveats=[L.escape_keep_math(c) for c in model.caveats],
    )


def _measured_peaks_sentence(o) -> str:
    """Quote the empirically measured peaks alongside the model verdict."""
    pks = getattr(o, "measured_peaks", None) or []
    if not pks:
        return ""
    bits = []
    for k, p in enumerate(pks, 1):
        err = f" $\\pm$ {_fmt(p['d_err'], 2)}" if p.get("d_err") else ""
        bits.append(f"peak {k}: $Q$ = {_fmt(p['q'], 3)} "
                    f"$\\mathrm{{\\AA}}^{{-1}}$, $d$ = {_fmt(p['d'], 4)}{err} "
                    f"$\\mathrm{{\\AA}}$ ({_fmt(p['d'] / 10, 3)} nm)")
    out = (" Independently of any sasmodels choice, an empirical fit "
           "(correlation-type background plus one Gaussian per peak) resolves "
           + "; ".join(bits) + " — see the peak-fit figure for this group.")
    return out + _peak_consistency_warning(o, pks)


def _peak_consistency_warning(o, pks) -> str:
    """Say so when the model's spacing does not match the measured peaks.

    A model can win on curve-shape agreement while reporting a spacing that
    corresponds to no observed peak. Quoting that number without comment would be
    the worst outcome of the whole analysis, so the disagreement is stated.
    """
    d = (getattr(o, "d_spacing", None) or {}).get("d")
    if not d or not pks:
        return ""
    measured = [p["d"] for p in pks if p.get("d")]
    # accept the model spacing if it matches a measured peak, or a small-integer
    # order of one (a 2nd-order reflection sits at d/2, and so on)
    for md in measured:
        for n in (1, 2, 3):
            if abs(d - md / n) / (md / n) < 0.12:
                return ""
    got = ", ".join(f"{m:.0f}" for m in measured)
    return (f" \\textbf{{Caution:}} the spacing this model reports "
            f"({_fmt(d, 4)} $\\mathrm{{\\AA}}$) does not correspond to any "
            f"measured peak ({got} $\\mathrm{{\\AA}}$), nor to a low-order "
            "reflection of one. The model has therefore fitted the curve without "
            "reproducing the peak structure, and its spacing should NOT be quoted "
            "as the repeat distance — use the measured peak positions above.")


def _dspacing_sentence(o) -> str:
    """State the repeat distance in the fit section when the model gives one."""
    d = getattr(o, "d_spacing", None) or {}
    if not d.get("d"):
        return ""
    err = f" $\\pm$ {_fmt(d['d_err'], 2)}" if d.get("d_err") else ""
    return (f" Repeat distance from the fitted peak: "
            f"$d = 2\\pi/Q_\\mathrm{{peak}}$ = {_fmt(d['d'], 4)}{err} "
            f"$\\mathrm{{\\AA}}$ ({_fmt(d['d'] / 10, 3)} nm), "
            f"from $Q_\\mathrm{{peak}}$ = {_fmt(d['q_peak'], 3)} "
            f"$\\mathrm{{\\AA}}^{{-1}}$.")


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
            f"{L.escape(str(a.get('verdict','?')).replace('_',' '))}"
            + (f", score={_fmt(a.get('shape_score'),2)}" if a.get('shape_score') is not None else "")
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
            "fitted_fixed": fitted_fixed + _dspacing_sentence(o)
                            + _measured_peaks_sentence(o),
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
    # Member and Condition often carry the same label (the group's independent
    # variable IS the sample name). Two identical columns only cost width.
    if rows and all(str(r[0]) == str(r[1]) for r in rows):
        headers = [headers[0]] + headers[2:]
        rows = [[r[0]] + r[2:] for r in rows]
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


def _sas_candidate_table(o):
    """Per-group candidate-comparison table: every model the fitter evaluated,
    ranked by shape score (higher = better), with reduced chi^2, the critic
    verdict, and any note (e.g. a parameter pinned at a bound)."""
    if not o.attempts:
        return None
    ok = [a for a in o.attempts if a.get("ok")]
    bad = [a for a in o.attempts if not a.get("ok")]
    ok.sort(key=lambda a: a.get("shape_score") or 0.0, reverse=True)
    rows = []
    for a in ok + bad:
        verdict = str(a.get("verdict", "—")).replace("_", " ")
        # prefer the physics penalties (they explain any down-ranking); else the note
        note = "; ".join(a.get("penalties") or []) or (a.get("note") or "")
        rows.append([
            a.get("model", "—"),
            _fmt(a.get("shape_score"), 3) if a.get("shape_score") is not None else "—",
            _fmt(a.get("reduced_chisq"), 3) if a.get("reduced_chisq") else "—",
            verdict,
            note[:90],
        ])
    return TableSpec(
        caption=f"Candidate models evaluated for {o.label}, ranked by shape score "
                "(higher is better). The shape score combines log-residual size, "
                "slope/curvature agreement, feature-position (knee/bump/valley) "
                "agreement, and residual randomness; the best-scoring model was chosen.",
        label=f"tab:sascand_{_safe(o.group_id)}",
        headers=["Model", "Shape score", "Reduced chi2", "Critic", "Note"],
        rows=rows, fontsize="footnotesize")


def _why_selected(o) -> str:
    """One-paragraph, deterministic explanation of why the reported model won."""
    if not o.best:
        return "No candidate model produced a usable fit for this group."
    win_name = o.best.model_name
    ok = [a for a in o.attempts if a.get("ok")]
    win = next((a for a in ok if a.get("model") == win_name), None)
    win_score = (win or {}).get("shape_score")
    runners = sorted((a for a in ok if a.get("model") != win_name),
                     key=lambda a: a.get("shape_score") or 0.0, reverse=True)
    s = f"Selected {win_name}"
    if win_score is not None:
        s += f": best shape score {_fmt(win_score, 2)}"
        if o.best.reduced_chisq:
            s += f" (reduced chi-squared {_fmt(o.best.reduced_chisq, 2)})"
    if runners:
        s += f", ahead of {runners[0].get('model')} ({_fmt(runners[0].get('shape_score'), 2)})"
        if len(runners) > 1:
            s += f" and {len(runners) - 1} other candidate(s)"
    s += (". Ranking is by a shape-aware score — log-residual size, slope/curvature "
          "agreement, and the positions of the knee/bump/valley features — so the "
          "chosen model reproduces the measured curve shape most faithfully, not "
          "merely the lowest chi-squared. ")
    s += ("The critic accepted this fit." if o.success
          else "The critic did not fully accept any candidate; this is the best available.")
    return s


def _build_sas_appendix(model: ReportModel, mode: str) -> list:
    if mode != "comprehensive" or not model.sas_fits:
        return []
    out = []
    for o in model.sas_fits:
        out.append({
            "title": L.escape(o.label or o.group_id),
            "why": L.escape(_why_selected(o)),
            "rationale": L.escape_keep_math(o.rationale or ""),
            "critique": L.escape_keep_math(o.critique or ""),
            "table": _sas_candidate_table(o),
        })
    return out


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
