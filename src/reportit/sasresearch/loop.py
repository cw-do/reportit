"""The autoresearch driver: observe -> hypothesise -> fit -> judge -> refine.

For one I(Q) curve:

  1. OBSERVE   — characterise the curve's shape (slopes, knee, bumps/valleys).
  2. HYPOTHESISE — pick candidate models (an LLM proposes them from the shape +
     the reference knowledge, or fall back to a default soft-matter shortlist).
  3. FIT       — each model from many starts (search.multistart_fit).
  4. JUDGE     — rank by the SHAPE-AWARE score (shape.shape_score). This is the
     arbiter; the LLM only adds a vision sanity-check on the top fits.
  5. REFINE    — try a couple of alternative fit windows on the leader and keep
     the best by the same score.

Outputs (all under out_dir): a fit plot per candidate, best_fit.png, a machine
-readable results.json, and notebook.md — a human "lab notebook" of what was
tried and why the winner won.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..analysis import knowledge, sascatalog
from ..analysis.loaders import load_iq
from ..analysis.sas_agent import _SELECT_SYS, _VISION_SYS
from ..plotting import figures
from . import search, shape

logger = logging.getLogger(__name__)

# Ornstein-Zernike (lorentz), correlation length, Teubner-Strey (microemulsion
# peak), and the polymer/Guinier family — the models a human reaches for on
# solution / soft-matter SANS.
DEFAULT_MODELS = [
    "correlation_length", "lorentz", "teubner_strey", "broad_peak",
    "guinier_porod", "mono_gauss_coil", "polymer_excl_volume", "dab",
]


def explore(datafile, out_dir, *, llm=None, context: str = "", models=None,
            n_starts: int = 8, use_llm: bool = True, refine: bool = True) -> dict:
    datafile = Path(datafile)
    out_dir = Path(out_dir)
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    iq = load_iq(datafile)
    desc = shape.curve_descriptors(iq.mod_q, iq.intensity)
    logger.info("curve: %d points, low-Q slope %s, knee %s, peaks %s",
                desc.get("n_points"), desc.get("low_q_slope"),
                desc.get("knee_q"), desc.get("peaks_q"))

    # --- pick candidate models ------------------------------------------- #
    llm_cands: dict = {}
    if models:
        model_names = list(models)
    elif use_llm and llm is not None:
        llm_cands = _llm_select(llm, context, desc, datafile.name)
        model_names = list(llm_cands) or list(DEFAULT_MODELS)
    else:
        model_names = list(DEFAULT_MODELS)
    # always compare a correlation/OZ description (needs no Guinier region)
    if not models:
        for m in ("correlation_length", "lorentz"):
            if m not in model_names:
                model_names.append(m)
    logger.info("candidate models: %s", model_names)

    # --- fit each -------------------------------------------------------- #
    ranked = []
    for name in model_names:
        cand = _prepare_cand(name, llm_cands.get(name))
        best, score, attempts = search.multistart_fit(
            iq, cand, name, desc, n_starts=n_starts)
        if best is None:
            ranked.append({"model": name, "ok": False, "score": 0.0,
                           "why": (llm_cands.get(name) or {}).get("why", ""),
                           "note": _first_note(attempts), "attempts": attempts})
            continue
        if refine:
            best, score, cand = _refine(iq, cand, name, desc, best, score)
        fig = cand_dir / f"{_safe(name)}.png"
        figures.plot_fit(best, fig, title=f"{name}  (score {score.get('score', 0):.3f})")
        ranked.append({
            "model": name, "ok": True,
            "score": round(score.get("score", 0.0), 4),
            "subscores": {k: round(score.get(k, 0.0), 3)
                          for k in ("resid", "chisq", "slope", "feature", "runs")},
            "reduced_chisq": best.reduced_chisq,
            "params": _params_pm(best),
            "window": [best.fit_qmin, best.fit_qmax],
            "note": best.note,
            "why": (llm_cands.get(name) or {}).get("why", ""),
            "figure": str(fig) if fig.is_file() else None,
            "attempts": attempts, "_result": best,
        })

    ranked.sort(key=lambda e: e.get("score", 0.0), reverse=True)

    # --- LLM vision sanity-check on the top fits (a check, not the judge) - #
    if use_llm and llm is not None:
        for entry in [e for e in ranked if e.get("figure")][:3]:
            try:
                entry["vision"] = llm.chat_vision(
                    _VISION_SYS, f"Model: {entry['model']}. Assess this fit.",
                    entry["figure"], cache_key=f"sasresearch:vision:{datafile.name}:{entry['model']}")
            except Exception as e:  # noqa: BLE001
                logger.debug("vision check failed for %s: %s", entry["model"], e)

    # --- write artefacts ------------------------------------------------- #
    winner = next((e for e in ranked if e.get("ok")), None)
    if winner and winner.get("_result") is not None:
        figures.plot_fit(winner["_result"], out_dir / "best_fit.png",
                         title=f"BEST: {winner['model']}  (score {winner['score']:.3f})")
    _write_results(out_dir / "results.json", datafile, desc, model_names, ranked)
    _write_notebook(out_dir / "notebook.md", datafile, context, desc, ranked, winner)
    logger.info("wrote %s", out_dir)
    return {"winner": winner["model"] if winner else None,
            "ranked": ranked, "descriptors": desc, "out_dir": str(out_dir)}


# --------------------------------------------------------------------------- #
# model selection + plan preparation
# --------------------------------------------------------------------------- #
def _prepare_cand(model_name: str, llm_cand: dict | None) -> dict:
    """Merge an LLM-proposed plan with the derived default so a fit always has a
    valid parameter/bounds set even when the LLM is terse or absent."""
    base = search.build_plan(model_name)
    if not llm_cand:
        return base
    fit = llm_cand.get("fit") or base["fit"]
    bounds = {**base["bounds"], **(llm_cand.get("bounds") or {})}
    return {
        "model": model_name, "fit": fit, "bounds": bounds,
        "initial": llm_cand.get("initial") or {},
        "q_min": llm_cand.get("q_min"), "q_max": llm_cand.get("q_max"),
    }


def _llm_select(llm, context, desc, fname) -> dict:
    catalog = sascatalog.short_catalog()
    cat = "\n".join(f"- {m['name']}: {m['description']} [params: {', '.join(m['parameters'])}]"
                    for m in catalog)
    kb = knowledge.load_knowledge()
    kb_block = f"REFERENCE KNOWLEDGE:\n{kb[:60000]}\n\n" if kb else ""
    prompt = (
        kb_block +
        f"EXPERIMENT CONTEXT:\n{context or '(none provided)'}\n\n"
        f"DATA FILE: {fname}\n\n"
        f"CURVE SHAPE (measured):\n{json.dumps(desc, default=str)}\n\n"
        f"AVAILABLE MODELS (priority first):\n{cat}\n\n"
        'Return JSON: {"candidates": [ {"model": <name>, "why": <reason>, '
        '"initial": {param: value}, "fit": [params to optimize], '
        '"bounds": {param: [lo, hi]}, "q_min": <number or null>, '
        '"q_max": <number or null>} ] }. Give 3-6 candidates, best first, '
        "spanning genuinely different physics (e.g. a correlation-length form, "
        "an Ornstein-Zernike/Lorentzian, a peak model, and a polymer form) so "
        "they can be compared.")
    try:
        plan = llm.chat_json(_SELECT_SYS, prompt, model=llm.settings.reasoning_model,
                             max_tokens=8000, cache_key=f"sasresearch:select:{fname}")
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM model selection failed (%s); using defaults", e)
        return {}
    out = {}
    for c in plan.get("candidates") or []:
        name = c.get("model")
        if name and name not in out:
            out[name] = c
    return out


# --------------------------------------------------------------------------- #
# deterministic window refinement (judged by the same shape score)
# --------------------------------------------------------------------------- #
def _refine(iq, cand, model_name, desc, best, best_score):
    """Try a couple of alternative fit windows on the leader and keep the best
    by shape score: (a) exclude a low-Q upturn at the knee, (b) the full range."""
    trials = []
    knee = desc.get("knee_q")
    low_slope = desc.get("low_q_slope")
    # a low-Q upturn (steep negative slope below the knee) is often aggregation
    # outside the length scale of interest — try excluding it
    if knee and (low_slope is None or low_slope < -1.5):
        c = dict(cand); c["q_min"] = float(knee)
        trials.append(c)
    if cand.get("q_min") is not None:                 # also try the full range
        c = dict(cand); c["q_min"] = None
        trials.append(c)

    for c in trials:
        r, s, _ = search.multistart_fit(iq, c, model_name, desc, n_starts=4, steps=200)
        if r is not None and s.get("score", 0.0) > best_score.get("score", 0.0) + 1e-3:
            best, best_score, cand = r, s, c
    return best, best_score, cand


# --------------------------------------------------------------------------- #
# output writers
# --------------------------------------------------------------------------- #
def _params_pm(result) -> dict:
    out = {}
    for p, v in (result.params or {}).items():
        u = (result.uncertainties or {}).get(p)
        out[p] = {"value": v, "error": u}
    return out


def _fmt_params(result) -> str:
    parts = []
    for p, v in (result.params or {}).items():
        u = (result.uncertainties or {}).get(p)
        parts.append(f"{p}={_sig(v)}±{_sig(u)}" if u else f"{p}={_sig(v)}")
    return ", ".join(parts)


def _write_results(path, datafile, desc, model_names, ranked):
    payload = {
        "datafile": str(datafile),
        "descriptors": desc,
        "models_tried": model_names,
        "candidates": [{k: v for k, v in e.items() if k != "_result"} for e in ranked],
    }
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))


def _write_notebook(path, datafile, context, desc, ranked, winner):
    L = ["# SAS autoresearch — fitting notebook", "",
         f"**Data file:** `{datafile}`  ", ]
    if context:
        L.append(f"**Context:** {context}  ")
    L += ["", "## 1. Observed curve shape", "",
          f"- points: {desc.get('n_points')}, "
          f"Q ∈ [{_sig(desc.get('q_min'))}, {_sig(desc.get('q_max'))}] Å⁻¹",
          f"- low-Q slope ≈ {_sig(desc.get('low_q_slope'))}, "
          f"high-Q slope ≈ {_sig(desc.get('high_q_slope'))}",
          f"- knee at Q ≈ {_sig(desc.get('knee_q'))} Å⁻¹"
          + (f"  (size ≈ {_sig(1.0/desc['knee_q'])} Å)" if desc.get("knee_q") else ""),
          f"- bumps at Q = {desc.get('peaks_q')}, valleys at Q = {desc.get('valleys_q')}",
          f"- low-Q plateau: {desc.get('low_q_plateau')}", "",
          "## 2. Candidates ranked by shape score", "",
          "| rank | model | score | resid | χ² | slope | feature | runs | red-χ²ᵥ |",
          "|-----:|-------|------:|------:|---:|------:|--------:|-----:|--------:|"]
    for k, e in enumerate(ranked, 1):
        if not e.get("ok"):
            L.append(f"| {k} | {e['model']} | — | | | | | | *{_short(e.get('note'))}* |")
            continue
        s = e.get("subscores", {})
        L.append(
            f"| {k} | **{e['model']}** | {e['score']:.3f} | {s.get('resid','')} | "
            f"{s.get('chisq','')} | {s.get('slope','')} | {s.get('feature','')} | "
            f"{s.get('runs','')} | {_sig(e.get('reduced_chisq'))} |")
    L += ["", "## 3. Winner", ""]
    if winner:
        r = winner["_result"]
        L += [f"**{winner['model']}** — shape score {winner['score']:.3f}.  ",
              f"Fitted window Q ∈ [{_sig(r.fit_qmin)}, {_sig(r.fit_qmax)}] Å⁻¹.  ",
              "", f"Parameters: {_fmt_params(r)}  ",
              (f"\nWhy chosen: {winner.get('why')}" if winner.get("why") else ""),
              (f"\nModel note: {r.note}" if r.note else "")]
        if winner.get("vision"):
            L += ["", f"Vision sanity-check: {winner['vision']}"]
    else:
        L.append("No candidate produced an acceptable fit.")
    L += ["", "## 4. Vision notes on top fits", ""]
    for e in ranked[:3]:
        if e.get("vision"):
            L.append(f"- **{e['model']}**: {e['vision']}")
    L += ["", "## 5. What was tried", "",
          f"Each model was fit from 1 data-anchored start plus sampled starts "
          f"(size swept around the knee, exponents over their range); the best "
          f"start by shape score is reported above. See `results.json` for every "
          f"start's parameters and score, and `candidates/` for per-model plots."]
    path.write_text("\n".join(x for x in L if x is not None))


# --------------------------------------------------------------------------- #
def _first_note(attempts):
    for a in attempts:
        if a.get("note"):
            return a["note"]
    return "all starts failed"


def _short(s, n=48):
    s = str(s or "")
    return s if len(s) <= n else s[:n] + "…"


def _sig(x, n=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:.{n}g}"


def _jsonable(o):
    if isinstance(o, Path):
        return str(o)
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))
