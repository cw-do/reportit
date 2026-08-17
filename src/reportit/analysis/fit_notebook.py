"""Write a Markdown "fitting notebook" for a whole report run.

Same spirit as the standalone ``reportit.sasresearch`` notebook, but aggregated
across every fitted group: for each group it records the observed curve shape,
the model-selection rationale, every candidate model ranked by its shape-aware
score, the winning fit (parameters +- uncertainty, window, critic note), and
the per-member trend table.

It can get long when there are many groups — that is by design; it is the audit
trail of what the fitter tried and why, kept next to the PDF report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def write(outcomes: list, out_path: Path) -> Path | None:
    """Write the fitting notebook for all group outcomes. Returns the path."""
    outcomes = [o for o in (outcomes or []) if o is not None]
    if not outcomes:
        return None
    from .. import __version__
    L = ["# SAS model-fitting notebook", "",
         f"Produced by reportit v{__version__}.", "",
         "Audit trail of the autoresearch fitter: candidate models, their "
         "shape-aware scores, and the chosen fit for each sample group. The "
         "shape score judges the fit the way a human does — log-residual size, "
         "curvature/slope agreement, feature (knee/bump/valley) positions, and "
         "residual randomness — not reduced chi-squared alone.", ""]
    n_ok = sum(1 for o in outcomes if o.success)
    L.append(f"**{len(outcomes)} group(s) fitted; {n_ok} accepted by the critic.**")
    L.append("")
    for o in outcomes:
        L.extend(_group_section(o))
    out_path = Path(out_path)
    out_path.write_text("\n".join(L))
    logger.info("fitting notebook -> %s", out_path)
    return out_path


def _group_section(o) -> list[str]:
    L = ["", "---", "", f"## {o.label or o.group_id}", ""]
    if o.dataset_name:
        L.append(f"Representative dataset: `{o.dataset_name}`  ")

    d = o.descriptors or {}
    if d:
        knee = d.get("knee_q")
        size = f"  (size ≈ {_g(1.0 / knee)} Å)" if knee else ""
        guin = d.get("has_guinier_region")
        guin_txt = ("Guinier plateau present" if guin
                    else "NO Guinier plateau (low-Q rising → size models discouraged)"
                    if guin is False else "Guinier region undetermined")
        L += ["", "**Observed shape:** "
              f"{d.get('n_points','?')} points, "
              f"Q ∈ [{_g(d.get('q_min'))}, {_g(d.get('q_max'))}] Å⁻¹; "
              f"low-Q slope ≈ {_g(d.get('low_q_slope'))}, "
              f"high-Q slope ≈ {_g(d.get('high_q_slope'))}; "
              f"knee at Q ≈ {_g(knee)} Å⁻¹{size}; "
              f"bumps {d.get('peaks_q') or '—'}, valleys {d.get('valleys_q') or '—'}; "
              f"{guin_txt}.  "]

    if o.rationale:
        L += ["", f"**Model-selection rationale:** {o.rationale}  "]

    # candidates ranked by shape score (ok first, then failures)
    ok = [a for a in o.attempts if a.get("ok")]
    bad = [a for a in o.attempts if not a.get("ok")]
    ok.sort(key=lambda a: a.get("shape_score", 0.0), reverse=True)
    if ok or bad:
        L += ["", "**Candidates tried:**", "",
              "| model | shape score | reduced χ²ᵥ | critic | note |",
              "|-------|------------:|------------:|--------|------|"]
        for a in ok:
            note = "; ".join(a.get("penalties") or []) or (a.get("note") or "")
            L.append(f"| {a.get('model','?')} | {_g(a.get('shape_score'))} | "
                     f"{_g(a.get('reduced_chisq'))} | {a.get('verdict','—')} | "
                     f"{_short(note)} |")
        for a in bad:
            L.append(f"| {a.get('model','?')} | — | — | fit failed | "
                     f"{_short(a.get('note'))} |")

    # winner
    L += ["", "**Chosen fit:** "]
    if o.best is not None:
        b = o.best
        L[-1] += (f"**{b.model_name}** — "
                  f"{'accepted' if o.success else 'best available (not accepted)'}, "
                  f"reduced χ²ᵥ = {_g(b.reduced_chisq)}, "
                  f"window Q ∈ [{_g(b.fit_qmin)}, {_g(b.fit_qmax)}] Å⁻¹.  ")
        L += ["", f"Parameters: {_params(b)}  "]
        if o.critique:
            L += ["", f"Critic assessment: {o.critique}"]
    else:
        L[-1] += "none — no candidate produced a usable fit."

    # per-member trend
    fits = o.member_fits or []
    if len(fits) >= 2:
        param = o.trend_param
        L += ["", f"**Per-member fits** (trend parameter: `{param or '—'}`):", "",
              "| condition | " + (f"{param} | " if param else "")
              + "reduced χ²ᵥ |",
              "|-----------|" + ("------:|" if param else "") + "-----------:|"]
        for f in fits:
            cell = f"| {f.get('condition','?')} | "
            if param:
                v = (f.get("params") or {}).get(param)
                u = (f.get("uncertainties") or {}).get(param)
                cell += (f"{_g(v)} ± {_g(u)} | " if u else f"{_g(v)} | ")
            cell += f"{_g(f.get('reduced_chisq'))} |"
            L.append(cell)
    return L


def _params(result) -> str:
    parts = []
    for p, v in (result.params or {}).items():
        u = (result.uncertainties or {}).get(p)
        parts.append(f"{p} = {_g(v)} ± {_g(u)}" if u else f"{p} = {_g(v)}")
    return ", ".join(parts) or "—"


def _g(x, n: int = 4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{x:.{n}g}" if np.isfinite(x) else "—"


def _short(s, n: int = 60) -> str:
    s = str(s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"
