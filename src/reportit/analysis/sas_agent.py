"""Agentic, model-based SANS fitting for one sample group.

Workflow (per group, on a representative extended-Q curve):
  1. model-selector agent (reasoning LLM): given the proposal context, the curve
     shape, and the sasmodels catalog, propose an ordered list of candidate
     models, each with a parameter plan (initial guesses, which to fit/fix,
     bounds).
  2. fit each candidate with sasmodels+bumps.
  3. critic: a vision LLM inspects the fit-vs-data plot, a reasoning LLM judges
     chi^2 / residuals / parameter sanity and decides accept | reject.
  4. stop when a fit is accepted (or candidates exhausted); keep the best.
Reports successes AND honest failures.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from ..llm import LLMClient
from ..models import Dataset, SasFitOutcome
from ..plotting import figures
from ..sasresearch import search, shape
from . import knowledge, sascatalog, sasfit
from .loaders import load_iq
from .metrics import analyze

logger = logging.getLogger(__name__)

# Multi-start counts for the autoresearch fitter: more starts for the
# representative curve that drives model selection, fewer per member (cost).
_N_STARTS = 6
_N_STARTS_MEMBER = 3

_SELECT_SYS = (
    "You are an expert small-angle neutron scattering (SANS) data analyst choosing "
    "SasView/sasmodels models to fit a measured I(Q) curve. Use the experiment "
    "context and the ACTUAL curve shape (provided as downsampled log-log points "
    "and slope features) to pick models that are physically appropriate. For each "
    "candidate give sensible initial guesses (e.g. estimate Rg or a correlation "
    "length from where the curve bends), choose which parameters to FIT vs keep "
    "FIXED (typically fit shape + scale + background; fix things the data can't "
    "constrain), and give physical bounds. Order candidates best-first; prefer the "
    "simplest model that captures the relevant physics.\n"
    "Partial-Q-range fitting is valid and encouraged WHEN appropriate: a model "
    "often describes only PART of the curve. A low-Q upturn is frequently "
    "aggregation/large-scale structure OUTSIDE the length scale of interest — set "
    "q_min to exclude ONLY that upturn. CRITICAL: do NOT cut into the form-factor "
    "knee / Guinier bend that constrains the size (Rg) — excluding it makes the fit "
    "worse and the size unconstrained. Put q_min just above the aggregation upturn, "
    "not above the knee. Use q_max to drop a noisy/background-dominated tail. When "
    "unsure where the regime boundary is, propose BOTH a full-range candidate "
    "(q_min/q_max null) and a restricted one so they can be compared. State why.\n"
    "GUINIER-REGION RULE (decisive — obey the measured curve, not expectations from "
    "the proposal): size-based models that report an Rg/radius — guinier, "
    "guinier_porod, mono_gauss_coil, poly_gauss_coil, polymer_excl_volume, and "
    "form factors (sphere, ellipsoid, cylinder, ...) — REQUIRE a resolved Guinier "
    "region: a low-Q PLATEAU where I(Q) flattens (log-log slope approaching 0) "
    "before the higher-Q power law. The CURVE FEATURES include has_guinier_region. "
    "If it is false (the low-Q keeps RISING — an upturn or unbroken power law with "
    "no flattening), the size is UNCONSTRAINED and such a model fails at low Q (it "
    "rolls over while the data shoots up). In that case DO NOT rely on size-based "
    "models; use correlation_length (Lorentzian + Porod) or lorentz "
    "(Ornstein-Zernike), which describe correlation/network scattering and fit the "
    "full Q-range without a Guinier plateau. Excluding the low-Q with q_min to prop "
    "up a size model is NOT acceptable — a correlation model that fits the full "
    "range is preferred. guinier_porod is appropriate ONLY for a genuine two-level "
    "system: globular objects whose Guinier knee IS within range PLUS separate "
    "large-scale structure of unknown size — not merely to absorb an upturn. For "
    "solution / semidilute samples, ALWAYS include correlation_length and lorentz "
    "among your candidates so they are compared."
)

_CRITIC_SYS = (
    "You are a rigorous but fair SANS fitting referee. Given a model, its fitted "
    "parameters, reduced chi-squared, R^2, the fitted Q-window, and a visual "
    "description of the fit-vs-data plot, decide whether the fit is acceptable AND "
    "USEFUL.\n"
    "Key principle: a fit that describes only a LIMITED Q-range can still be valid "
    "and informative. Do NOT reject a model solely because it misses a low-Q upturn "
    "or a high-Q tail that lies OUTSIDE its fitted window — if that deviation is "
    "plausibly due to out-of-scope structure (aggregation, large-scale "
    "correlations) or background, ACCEPT the fit and note its range of validity. "
    "Within the FITTED window, judge whether the model captures the shape, whether "
    "residuals are random vs systematic, and whether parameters are physical and "
    "well-constrained. Reject only when the model is wrong for the regime it claims "
    "to describe (systematic misfit inside the fitted window, or unphysical "
    "parameters).\n"
    "WEIGHTING: in SANS, reduced chi-squared and R^2 are necessary but NOT "
    "sufficient — the VISUAL assessment (does the fitted line follow the data, are "
    "residuals random?) is the primary criterion. A visually faithful fit with "
    "moderate chi-squared can be acceptable; a low-chi-squared fit with systematic "
    "residuals or a parameter pinned at a bound should be rejected.\n"
    "REJECT (regardless of chi^2) when: (a) the fitted line clearly departs from "
    "the data markers at low Q — most often a size-based model (guinier_porod, "
    "polymer_excl_volume, gauss_coil, form factors) that ROLLS OVER while the data "
    "keeps RISING, or a fit that only 'works' by excluding the low-Q upturn and "
    "then diverges from those excluded points; (b) a size-based/chain model is used "
    "but the data has NO Guinier plateau (no low-Q flattening), so the reported Rg "
    "is meaningless; (c) a size parameter (Rg, radius, cor_length) is pinned at a "
    "bound or has a relative uncertainty > ~1 (unconstrained). In these cases say a "
    "correlation_length / Lorentzian (Ornstein-Zernike) description is the "
    "physically appropriate choice."
)

_VISION_SYS = (
    "You are inspecting a SANS fit-vs-data plot (log-log I(Q) on top, fractional "
    "residuals below). Describe concisely how well the red model curve follows the "
    "blue data points across Q, and whether residuals are random or show "
    "systematic structure (e.g. model misses the low-Q plateau or high-Q slope)."
)


def _curve_features(ds: Dataset) -> dict:
    path = ds.merged_path or ds.iq_path
    iq = load_iq(path)
    q = np.asarray(iq.mod_q, float)
    i = np.asarray(iq.intensity, float)
    m = (q > 0) & (i > 0) & np.isfinite(q) & np.isfinite(i)
    q, i = q[m], i[m]
    order = np.argsort(q)
    q, i = q[order], i[order]
    n = len(q)
    idx = np.linspace(0, n - 1, min(n, 24)).astype(int)
    da = analyze(ds.output_name, ds.variant, path)
    gd = shape.curve_descriptors(q, i)
    return {
        "path": str(path),
        "q_min": float(q.min()), "q_max": float(q.max()), "n_points": n,
        "low_q_slope": da.low_q_slope, "high_q_slope": da.high_q_slope,
        # shape cues the selector must respect (esp. the Guinier-region flag)
        "has_guinier_region": gd.get("has_guinier_region"),
        "knee_q": gd.get("knee_q"),
        "low_q_plateau": gd.get("low_q_plateau"),
        "peaks_q": gd.get("peaks_q"), "valleys_q": gd.get("valleys_q"),
        "loglog_points": [[round(float(q[k]), 5), round(float(i[k]), 5)] for k in idx],
    }


def _select_prompt(context: str, group_label: str, feats: dict, catalog: list) -> str:
    cat = "\n".join(f"- {m['name']}: {m['description']} [params: {', '.join(m['parameters'])}]"
                    for m in catalog)
    kb = knowledge.load_knowledge(stage="fitting",
                                  context=f"{context}\n{json.dumps(feats, default=str)}")
    kb_block = f"REFERENCE KNOWLEDGE (general SANS model-selection guidance):\n{kb}\n\n" if kb else ""
    return (
        kb_block +
        f"EXPERIMENT CONTEXT:\n{context}\n\n"
        f"GROUP: {group_label}\n\n"
        f"CURVE FEATURES:\n{json.dumps(feats, default=str)}\n\n"
        f"AVAILABLE MODELS (priority first):\n{cat}\n\n"
        'Return JSON: {"candidates": [ {"model": <name>, "why": <reason>, '
        '"initial": {param: value}, "fit": [params to optimize], '
        '"fixed": {param: value}, "bounds": {param: [lo, hi]}, '
        '"q_min": <number or null>, "q_max": <number or null>} ], '
        '"ordering_rationale": <text> }. Give 1-3 candidates, best first. '
        "DEFAULT q_min and q_max to null (fit the FULL measured range). Only set "
        "q_min for a genuine out-of-scope low-Q upturn the model cannot describe "
        "(e.g. aggregation), or q_max to drop a noisy tail — never trim low-Q just "
        "to lower chi^2. A correlation-type model with a Porod term can usually fit "
        "the low-Q upturn directly, so it should use the full range."
    )


def run_group_fit(
    group, members: list[Dataset], llm: LLMClient, fig_dir: Path,
    experiment_context: str, *, max_models: int = 3, guide=None, namemap=None,
    d_intent=None, m_intent=None,
) -> SasFitOutcome:
    glabel = namemap.shorten_label(group.label) if namemap is not None else group.label
    out = SasFitOutcome(group_id=group.group_id, label=glabel)
    rep = _representative(members)
    if rep is None:
        out.critique = "No fittable dataset in group."
        return out
    out.dataset_name = rep.output_name

    try:
        feats = _curve_features(rep)
    except Exception as e:  # noqa: BLE001
        out.critique = f"Could not load curve: {e}"
        return out

    catalog = sascatalog.short_catalog()
    reasoning = llm.settings.reasoning_model
    from .. import guidance
    fit_hint = guidance.hint_block(guide, "fitting")
    gkey = guidance.digest(guide)
    from . import dspacing
    d_hint = dspacing.prompt_hint(d_intent) if d_intent is not None else ""
    from . import model_intent as _mi
    d_hint += _mi.prompt_hint(m_intent) if m_intent is not None else ""
    dkey = "d" if d_hint else "n"
    kbkey = knowledge.digest("fitting")

    # 1) model selection
    try:
        plan = llm.chat_json(_SELECT_SYS + d_hint + fit_hint,
                             _select_prompt(experiment_context, group.label, feats, catalog),
                             model=reasoning, max_tokens=8000,
                             cache_key=f"sasselect:{group.group_id}:{rep.output_name}:g={gkey}:{dkey}:"
                                       f"kb={kbkey}")
    except Exception as e:  # noqa: BLE001
        out.critique = f"Model selection failed: {e}"
        return out
    candidates = plan.get("candidates") or []
    out.rationale = plan.get("ordering_rationale", "")
    if not candidates:
        out.critique = "Selector proposed no candidate models."
        return out
    # always compare a correlation/OZ description (needs no Guinier region)
    candidates = _ensure_baseline(candidates[:max_models], d_intent)

    # 2) fit EVERY candidate with the robust multi-start fitter and rank by the
    # PHYSICS-ADJUSTED shape score (no LLM here — the score is the judge). We keep
    # a full comparison so the report can show what was tried and why one won.
    best = None
    best_cand = None
    best_fig = None
    iq = load_iq(feats["path"])
    data_desc = _data_desc(iq)
    out.descriptors = data_desc
    scored = []  # (result, cand, attempt, fig_path)
    for cand in candidates:
        model_name = cand.get("model")
        if not model_name:
            continue
        result = _fit_robust(iq, cand, model_name, data_desc, n_starts=_N_STARTS)
        if result is None or not result.ok:
            out.attempts.append({
                "model": model_name, "ok": False, "verdict": "fit_failed",
                "note": (result.note if result else "all starts failed"),
                "params": (result.params if result else {})})
            continue
        fig_path = fig_dir / f"sasfit_{_safe(group.group_id)}_{_safe(model_name)}.png"
        figures.plot_fit(result, fig_path, title=f"{glabel}: {model_name}")
        sc = getattr(result, "_shape_score", None) or {}
        attempt = {"model": model_name, "ok": True, "verdict": "not selected",
                   "reduced_chisq": result.reduced_chisq, "r2": result.r_squared,
                   "shape_score": round(_score_of(result), 4),
                   "base_score": sc.get("base_score"), "penalties": sc.get("penalties", []),
                   "note": result.note, "params": result.params}
        out.attempts.append(attempt)
        scored.append((result, cand, attempt, fig_path))
        if best is None or _score_of(result) > _score_of(best):
            best, best_cand, best_fig = result, cand, fig_path

    if best is None:
        out.critique = out.critique or "All candidate fits failed."
        return out

    out.best = best
    out.figure = _figref(best_fig, glabel, best.model_name, best)

    # 3) LLM check on the WINNER only (vision + reasoning critic). The score chose
    # the model; the critic confirms it visually and flags any physical problems.
    win_attempt = next(a for (_r, _c, a, _f) in scored if _r is best)
    vision_note = ""
    if best_fig.is_file():
        vision_note = llm.chat_vision(
            _VISION_SYS, f"Model: {best.model_name}. Assess this fit.", best_fig,
            cache_key=f"sasvision:{group.group_id}:{best.model_name}")
    verdict = _critique(llm, reasoning, group.label, best, vision_note,
                        experiment_context, fit_hint=fit_hint, gkey=gkey)
    out.success = bool(verdict.get("accept"))
    out.critique = verdict.get("assessment", "")
    win_attempt["verdict"] = "accepted" if out.success else "rejected"
    win_attempt["quality"] = verdict.get("quality")
    win_attempt["critique"] = verdict.get("assessment", "")
    win_attempt["vision"] = vision_note

    # Prefer the FULL measured Q-range: if the winner was fit over a restricted
    # (low-Q-excluded) window, refit full-range and adopt it unless clearly worse.
    try:
        _prefer_full_coverage(out, iq, best_cand, out.best.model_name, data_desc,
                              group, fig_dir, label=glabel)
    except Exception as e:  # noqa: BLE001
        logger.debug("full-coverage step failed for %s: %s", group.group_id, e)

    # Independent, model-free peak measurement for THIS curve. Recorded so the
    # fitting section can state where the peaks actually are — a report that says
    # a model "fails to capture the lamellar peak" must also say where that peak
    # is, otherwise the reader is told about a failure and given no number.
    if d_intent is not None and getattr(d_intent, "wanted", False):
        try:
            mp = dspacing.fit_peak_model(iq.mod_q, iq.intensity,
                                         getattr(iq, "error", None),
                                         q_window=d_intent.q_window)
            if mp.ok and mp.peaks:
                out.measured_peaks = [
                    {"q": p_.q, "q_err": p_.q_err, "d": p_.d, "d_err": p_.d_err,
                     "width_q": p_.width_q} for p_ in mp.peaks]
        except Exception as e:  # noqa: BLE001
            logger.debug("empirical peak measurement failed: %s", e)

    # repeat distance from the fitted peak position, when the model has one
    pk = dspacing.from_fit(out.best)
    if pk is not None:
        out.d_spacing = {"q_peak": pk.q, "d": pk.d, "d_err": pk.d_err,
                         "method": pk.method}
        logger.info("  repeat distance from fit: d = %.1f A (%.1f nm) [%s]",
                    pk.d, pk.d / 10, pk.method)

    # model description / equation for the report
    det = sascatalog.model_detail(out.best.model_name)
    if det:
        out.model_description = det.get("description", "")

    # Fit EVERY member with the chosen model so we can report a trend
    # (e.g. Rg vs temperature). Pure bumps fits — no extra LLM calls.
    try:
        _fit_all_members(out, group, members, best.model_name, best_cand, fig_dir,
                         namemap=namemap)
    except Exception as e:  # noqa: BLE001
        logger.warning("per-member fitting failed for %s: %s", group.group_id, e)
    return out


# --------------------------------------------------------------------------- #
# fit policy + helpers
# --------------------------------------------------------------------------- #
def _data_desc(iq) -> dict:
    """Shape descriptors (knee, slopes, features, Guinier flag) for the fitter."""
    return shape.curve_descriptors(iq.mod_q, iq.intensity)


def _ensure_baseline(candidates: list, d_intent=None) -> list:
    """Guarantee the always-relevant models are among the candidates.

    correlation_length and lorentz: a correlation / Ornstein-Zernike description
    needs no Guinier region and fits the full Q-range, so it must never be missing
    from the comparison.

    When the experiment targets a REPEAT DISTANCE, also force models that actually
    report a peak position or spacing — otherwise the report can end up saying a
    model "fails to capture the lamellar peak" without ever having tried a model
    that could.
    """
    from . import dspacing
    out = list(candidates)
    have = {(c.get("model") or "").lower() for c in out}
    forced = ["correlation_length", "lorentz"]
    if d_intent is not None and getattr(d_intent, "wanted", False):
        forced += list(dspacing.FORCE_MODELS)
    for m in forced:
        if m not in have:
            try:
                out.append(search.build_plan(m))
                have.add(m)
            except Exception as e:  # noqa: BLE001
                logger.debug("could not build a plan for %s: %s", m, e)
    return out


def _fit_robust(iq, cand, model_name, data_desc, *, n_starts):
    """Robust multi-start fit: many diverse starting points, ranked by the
    shape-aware score (stashed on the result). None if every start failed."""
    best, score, _ = search.multistart_fit(iq, cand, model_name, data_desc,
                                            n_starts=n_starts)
    if best is not None:
        best._shape_score = score
    return best


def _score_of(result) -> float:
    if result is None:
        return 0.0
    return float((getattr(result, "_shape_score", None) or {}).get("score", 0.0) or 0.0)


def _member_condition(group, ds) -> str:
    """Label a member by the group's INDEPENDENT VARIABLE, not a fixed field."""
    kind = (group.kind or "").lower()
    if "temperature" in kind:
        return ds.temperature or "RT"
    if "concentration" in kind:
        return ds.base or ds.output_name           # sample id stands in for concentration
    if "config" in kind:
        return ds.config or ds.output_name
    return ds.base or ds.output_name


def _member_order(group, ds):
    """Numeric ordering value for the trend x-axis (None => categorical)."""
    if "temperature" in (group.kind or "").lower():
        return _temp_value(ds.temperature)
    return None


def _prefer_full_coverage(out, iq, best_cand, model_name, data_desc, group,
                          fig_dir, *, margin: float = 0.03,
                          label: str | None = None) -> None:
    """Prefer fitting the FULL measured Q-range. If the winning model was fit over
    a RESTRICTED window (low-Q excluded), refit it over the full range and adopt
    that unless it is CLEARLY worse (shape score drops by more than `margin`).

    Rationale: excluding low-Q should be reserved for genuinely out-of-scope
    structure (e.g. an aggregation upturn the model cannot describe). When the
    model already describes the excluded points (its extrapolation tracks them),
    trimming them just to shave chi^2 is dishonest — report the full-range fit."""
    cur = out.best
    if cur is None or not cur.q_excluded:
        return  # already full range (or nothing fit)
    # was the exclusion at LOW Q? (high-Q trimming of a noisy tail can be kept)
    if cur.fit_qmin is None or cur.fit_qmin <= (data_desc.get("q_min") or 0) * 1.05:
        return
    cand_full = dict(best_cand or {})
    cand_full["q_min"] = None                    # undo the low-Q exclusion
    r_full = _fit_robust(iq, cand_full, model_name, data_desc, n_starts=_N_STARTS)
    if r_full is None or not r_full.ok:
        return
    if _score_of(r_full) >= _score_of(cur) - margin:
        fig = fig_dir / f"sasfit_{_safe(group.group_id)}_{_safe(model_name)}_full.png"
        figures.plot_fit(r_full, fig, title=f"{label or group.label}: {model_name} (full Q-range)")
        out.best = r_full
        out._best_cand = cand_full               # members inherit the full window
        out.figure = _figref(fig, label or group.label, model_name, r_full)
        out.critique += (
            f" Fit extended to the full measured Q-range "
            f"[{r_full.fit_qmin:.4g}, {r_full.fit_qmax:.4g}] "
            r"$\mathrm{\AA}^{-1}$ — the model describes the low-Q points, so no "
            "exclusion was warranted.")


def _fit_all_members(out, group, members, model_name, cand, fig_dir,
                     namemap=None) -> None:
    glabel = out.label or group.label      # already shortened by the caller
    cand = getattr(out, "_best_cand", None) or cand  # use refined window if any
    primary = _primary_param(cand, model_name)
    out.trend_param = primary or ""
    fits = []
    results = []  # (condition_label, SasFitResult) for the combined overlay plot
    seen_paths: set = set()
    for ds in members:
        path = ds.merged_path or ds.iq_path  # always prefer merged
        if not path or not Path(path).is_file():
            continue
        rp = str(Path(path).resolve())
        if rp in seen_paths:  # two configs share one merged file — fit it once
            continue
        seen_paths.add(rp)
        try:
            iq = load_iq(path)
            r = _fit_robust(iq, cand, model_name, _data_desc(iq),
                            n_starts=_N_STARTS_MEMBER)
        except Exception as e:  # noqa: BLE001
            logger.debug("member fit failed %s: %s", ds.output_name, e)
            continue
        if r is None or not r.ok:
            continue
        cond_val = _member_order(group, ds)
        cond_label = _member_condition(group, ds)
        if namemap is not None:
            cond_label = namemap.short(cond_label)   # short labels in fit plots
        results.append((cond_label, r))
        fits.append({
            # short display name: this is what the report's Member column and the
            # notebook show, and full output names make that table overflow the page
            "name": namemap.short(ds.output_name) if namemap is not None else ds.output_name,
            "condition": cond_label,
            "condition_val": cond_val,
            "params": r.params, "uncertainties": r.uncertainties,
            "reduced_chisq": r.reduced_chisq,
            "d_spacing": (lambda p: p.d if p else None)(_dspacing_of(r)),
        })
    # order sensibly: numerically by condition value when available (temperature),
    # else naturally by the condition label (sample id for concentration series).
    def _natkey(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

    def _fkey(f):
        return (0, f["condition_val"]) if f["condition_val"] is not None else (1, _natkey(f["condition"]))
    fits.sort(key=_fkey)
    results.sort(key=lambda lr: (0, _temp_value(lr[0])) if _temp_value(lr[0]) is not None
                 else (1, _natkey(lr[0])))
    out.member_fits = fits

    # Combined plot: ALL member fits in one figure (data markers + model lines).
    if len(results) >= 2:
        cfig = fig_dir / f"sasfit_{_safe(group.group_id)}_allfits.png"
        made = figures.plot_group_fits(
            f"{glabel}: {model_name} fits (all members)", results, cfig)
        if made:
            from ..models import FigureRef
            out.figure = FigureRef(
                path=made,
                caption=(f"Fits of all members of {glabel} to the {model_name} "
                         f"model (markers: data, lines: fitted model; faint x: points "
                         f"excluded from the fit window)."),
                label=f"fig:sasfit_{_safe(group.group_id)}")

    # trend figure for the primary parameter
    if primary and len(fits) >= 2:
        have = [f for f in fits if primary in (f["params"] or {})]
        numeric = [f for f in have if f["condition_val"] is not None]
        use_numeric = len(numeric) >= 2
        src = numeric if use_numeric else have
        points = [(f["condition_val"], f["params"].get(primary),
                   (f["uncertainties"] or {}).get(primary, 0), str(f["condition"]))
                  for f in src]
        if len(points) >= 2:
            kind = (group.kind or "").lower()
            if use_numeric:
                xlabel = "Temperature (C)" if "temperature" in kind else "condition"
            else:
                xlabel = "Sample" if "concentration" in kind else "series member"
            fig_path = fig_dir / f"trend_{_safe(group.group_id)}_{_safe(primary)}.png"
            made = figures.plot_trend(glabel, primary, points, fig_path,
                                      xlabel=xlabel, numeric_x=use_numeric)
            if made:
                from ..models import FigureRef
                out.trend_figure = FigureRef(
                    path=made,
                    caption=(f"Trend of fitted {primary} (from the {model_name} model) "
                             f"across {glabel}."),
                    label=f"fig:trend_{_safe(group.group_id)}")


def _dspacing_of(result):
    from . import dspacing
    return dspacing.from_fit(result)

def _primary_param(cand, model_name) -> str | None:
    fit_params = [p for p in ((cand or {}).get("fit") or [])
                  if p.lower() not in ("scale", "background", "bkg")]
    for pref in ("rg", "xi", "cor_length", "correlation_length", "radius",
                 "length", "i_zero", "rg_1", "lorentz_scale"):
        if pref in fit_params:
            return pref
    return fit_params[0] if fit_params else None


def _temp_value(temp):
    if not temp:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(temp))
    return float(m.group()) if m else None


def _critique(llm, reasoning, label, result, vision_note, context,
              fit_hint: str = "", gkey: str = "none") -> dict:
    payload = {
        "group": label,
        "model": result.model_name,
        "fitted_params": result.params,
        "uncertainties": result.uncertainties,
        "fixed_params": result.fixed,
        "reduced_chisq": result.reduced_chisq,
        "r_squared": result.r_squared,
        "fitted_q_window": [result.fit_qmin, result.fit_qmax],
        "n_points_excluded_low_or_high_q": len(result.q_excluded),
        "visual_assessment": vision_note,
        "experiment_context": context[:8000],
    }
    kb = knowledge.load_knowledge(stage="critic",
                                  context=f"{result.model_name} {context}")
    kb_block = f"\n\nReference SANS knowledge:\n{kb}\n" if kb else ""
    sys = _CRITIC_SYS + kb_block + fit_hint + (
        '\nReturn JSON: {"accept": bool, "quality": "good|fair|poor", '
        '"assessment": <2-3 sentence verdict>, "issues": [<strings>]}.')
    try:
        return llm.chat_json(sys, json.dumps(payload, default=str),
                             model=reasoning, max_tokens=4000,
                             cache_key=f"sascritic:{label}:{result.model_name}:"
                                       f"{round(result.reduced_chisq or 0,2)}:g={gkey}:"
                                       f"kb={knowledge.digest('critic')}")
    except Exception as e:  # noqa: BLE001
        logger.warning("critic failed: %s", e)
        return {"accept": (result.reduced_chisq or 1e9) < 5, "quality": "fair",
                "assessment": "Automated critic unavailable; judged on chi^2.", "issues": []}


def _representative(members: list[Dataset]) -> Dataset | None:
    for m in members:
        if m.merged_path:
            return m
    for m in members:
        if m.iq_path:
            return m
    return None


def _figref(path, label, model_name, result):
    from ..models import FigureRef
    cap = f"Model-based fit of {label} to the {model_name} model"
    if result.reduced_chisq:
        cap += f" (reduced $\\chi^2$={result.reduced_chisq:.1f})"
    if result.q_excluded and result.fit_qmin is not None:
        cap += (f". Fitted over Q=[{result.fit_qmin:.4g}, {result.fit_qmax:.4g}] "
                f"$\\mathrm{{\\AA}}^{{-1}}$; gray points lie outside this range")
    cap += "."
    return FigureRef(path=path, caption=cap, label=f"fig:sasfit_{_safe(label)}")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))
