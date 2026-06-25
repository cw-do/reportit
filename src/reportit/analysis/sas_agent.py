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
from . import sascatalog, sasfit
from .loaders import load_iq
from .metrics import analyze

logger = logging.getLogger(__name__)

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
    "(q_min/q_max null) and a restricted one so they can be compared. State why."
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
    "parameters)."
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
    return {
        "path": str(path),
        "q_min": float(q.min()), "q_max": float(q.max()), "n_points": n,
        "low_q_slope": da.low_q_slope, "high_q_slope": da.high_q_slope,
        "loglog_points": [[round(float(q[k]), 5), round(float(i[k]), 5)] for k in idx],
    }


def _select_prompt(context: str, group_label: str, feats: dict, catalog: list) -> str:
    cat = "\n".join(f"- {m['name']}: {m['description']} [params: {', '.join(m['parameters'])}]"
                    for m in catalog)
    return (
        f"EXPERIMENT CONTEXT:\n{context}\n\n"
        f"GROUP: {group_label}\n\n"
        f"CURVE FEATURES:\n{json.dumps(feats, default=str)}\n\n"
        f"AVAILABLE MODELS (priority first):\n{cat}\n\n"
        'Return JSON: {"candidates": [ {"model": <name>, "why": <reason>, '
        '"initial": {param: value}, "fit": [params to optimize], '
        '"fixed": {param: value}, "bounds": {param: [lo, hi]}, '
        '"q_min": <number or null>, "q_max": <number or null>} ], '
        '"ordering_rationale": <text> }. Give 1-3 candidates, best first. '
        "Use q_min/q_max to restrict the fit to the Q-range the model applies to "
        "(e.g. exclude a low-Q aggregation upturn)."
    )


def run_group_fit(
    group, members: list[Dataset], llm: LLMClient, fig_dir: Path,
    experiment_context: str, *, max_models: int = 3,
) -> SasFitOutcome:
    out = SasFitOutcome(group_id=group.group_id, label=group.label)
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

    # 1) model selection
    try:
        plan = llm.chat_json(_SELECT_SYS,
                             _select_prompt(experiment_context, group.label, feats, catalog),
                             model=reasoning, max_tokens=4000,
                             cache_key=f"sasselect:{group.group_id}:{rep.output_name}")
    except Exception as e:  # noqa: BLE001
        out.critique = f"Model selection failed: {e}"
        return out
    candidates = plan.get("candidates") or []
    out.rationale = plan.get("ordering_rationale", "")
    if not candidates:
        out.critique = "Selector proposed no candidate models."
        return out

    best = None
    best_cand = None
    iq = load_iq(feats["path"])
    for cand in candidates[:max_models]:
        model_name = cand.get("model")
        if not model_name:
            continue
        result = sasfit.fit_curve(
            iq.mod_q, iq.intensity, iq.error,
            model_name=model_name,
            initial=cand.get("initial") or {},
            fit_params=cand.get("fit") or [],
            bounds=cand.get("bounds") or {},
            q_min=cand.get("q_min"), q_max=cand.get("q_max"),
        )
        attempt = {"model": model_name, "ok": result.ok,
                   "reduced_chisq": result.reduced_chisq, "r2": result.r_squared,
                   "note": result.note, "params": result.params}
        if not result.ok:
            attempt["verdict"] = "fit_failed"
            out.attempts.append(attempt)
            continue

        fig_path = fig_dir / f"sasfit_{_safe(group.group_id)}_{_safe(model_name)}.png"
        figures.plot_fit(result, fig_path, title=f"{group.label}: {model_name}")

        # 3) critic — vision + reasoning
        vision_note = ""
        if fig_path.is_file():
            vision_note = llm.chat_vision(
                _VISION_SYS, f"Model: {model_name}. Assess this fit.", fig_path,
                cache_key=f"sasvision:{group.group_id}:{model_name}")
        verdict = _critique(llm, reasoning, group.label, result, vision_note,
                            experiment_context)
        attempt["verdict"] = "accept" if verdict.get("accept") else "reject"
        attempt["quality"] = verdict.get("quality")
        attempt["critique"] = verdict.get("assessment", "")
        attempt["vision"] = vision_note
        out.attempts.append(attempt)

        # track best by accept-then-lowest-chisq
        if best is None or _better(result, verdict, best, out):
            best = result
            best_cand = cand
            out.best = result
            out.figure = _figref(fig_path, group.label, model_name, result)
            out.critique = verdict.get("assessment", "")
            out.success = bool(verdict.get("accept"))

        if verdict.get("accept"):
            break

    if out.best is None:
        out.critique = out.critique or "All candidate fits failed."
        return out

    # Fit EVERY member with the chosen model so we can report a trend
    # (e.g. Rg vs temperature). Pure bumps fits — no extra LLM calls.
    try:
        _fit_all_members(out, group, members, best.model_name, best_cand, fig_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("per-member fitting failed for %s: %s", group.group_id, e)
    return out


def _fit_all_members(out, group, members, model_name, cand, fig_dir) -> None:
    primary = _primary_param(cand, model_name)
    out.trend_param = primary or ""
    fits = []
    for ds in members:
        path = ds.merged_path or ds.iq_path
        if not path or not Path(path).is_file():
            continue
        try:
            iq = load_iq(path)
            r = sasfit.fit_curve(
                iq.mod_q, iq.intensity, iq.error, model_name=model_name,
                initial=(cand or {}).get("initial") or {},
                fit_params=(cand or {}).get("fit") or [],
                bounds=(cand or {}).get("bounds") or {},
                q_min=(cand or {}).get("q_min"), q_max=(cand or {}).get("q_max"))
        except Exception as e:  # noqa: BLE001
            logger.debug("member fit failed %s: %s", ds.output_name, e)
            continue
        if not r.ok:
            continue
        cond_val = _temp_value(ds.temperature)
        fits.append({
            "name": ds.output_name,
            "condition": ds.temperature or "RT",
            "condition_val": cond_val,
            "params": r.params, "uncertainties": r.uncertainties,
            "reduced_chisq": r.reduced_chisq,
        })
    out.member_fits = fits

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
            xlabel = "Temperature (C)" if use_numeric else "sample"
            fig_path = fig_dir / f"trend_{_safe(group.group_id)}_{_safe(primary)}.png"
            made = figures.plot_trend(group.label, primary, points, fig_path,
                                      xlabel=xlabel, numeric_x=use_numeric)
            if made:
                from ..models import FigureRef
                out.trend_figure = FigureRef(
                    path=made,
                    caption=(f"Trend of fitted {primary} (from the {model_name} model) "
                             f"across {group.label}."),
                    label=f"fig:trend_{_safe(group.group_id)}")


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


def _critique(llm, reasoning, label, result, vision_note, context) -> dict:
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
        "experiment_context": context[:1500],
    }
    sys = _CRITIC_SYS + (
        '\nReturn JSON: {"accept": bool, "quality": "good|fair|poor", '
        '"assessment": <2-3 sentence verdict>, "issues": [<strings>]}.')
    try:
        return llm.chat_json(sys, json.dumps(payload, default=str),
                             model=reasoning, max_tokens=1200,
                             cache_key=f"sascritic:{label}:{result.model_name}:{round(result.reduced_chisq or 0,2)}")
    except Exception as e:  # noqa: BLE001
        logger.warning("critic failed: %s", e)
        return {"accept": (result.reduced_chisq or 1e9) < 5, "quality": "fair",
                "assessment": "Automated critic unavailable; judged on chi^2.", "issues": []}


def _better(result, verdict, best, out) -> bool:
    # prefer accepted fits; among same acceptance, lower reduced chisq
    new_acc = bool(verdict.get("accept"))
    old_acc = out.success
    if new_acc != old_acc:
        return new_acc
    rc_new = result.reduced_chisq if result.reduced_chisq is not None else 1e18
    rc_old = best.reduced_chisq if best.reduced_chisq is not None else 1e18
    return rc_new < rc_old


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
