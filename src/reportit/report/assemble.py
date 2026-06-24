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
    colspec = "l" * ncol
    head = " & ".join(f"\\textbf{{{L.escape(h)}}}" for h in table.headers)
    body_lines = []
    for row in table.rows:
        cells = [L.escape(c) for c in row]
        # pad/truncate to header width
        cells = (cells + [""] * ncol)[:ncol]
        body_lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[H]\n\\centering\n\\small\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\caption{{{L.escape_keep_math(table.caption)}}}\n"
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

    return template.render(
        mode=mode,
        title=L.escape(model.title),
        generated_at=L.escape(model.generated_at),
        overview=L.escape_keep_math(model.overview),
        science_goals=[L.escape_keep_math(g) for g in
                       (model.context.proposal.science_goals if model.context.proposal else [])],
        catalog_table=model.catalog_table,
        group_reports=groups,
        hypothesis_checks=hyp,
        discussion=L.escape_keep_math(model.discussion),
        caveats=[L.escape_keep_math(c) for c in model.caveats],
    )


def write_tex(model: ReportModel, out_dir: Path, mode: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = render(model, mode=mode)
    path = out_dir / f"report_{mode}.tex"
    path.write_text(tex)
    return path
