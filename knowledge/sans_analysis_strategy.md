# SANS 1D I(Q) analysis & model-selection strategy

Curated reference for choosing and fitting models to reduced 1D small-angle
scattering data. Distilled from three sources (full PDFs in `knowledge/sources/`
— consult them for derivations and worked examples):
- **Hammouda**, *A Tutorial on SANS from Polymers* (NIST) → `sources/tutorial_polymer.pdf`
- **Wei & Hore**, *Characterizing polymer structure with SANS* (J. Appl. Phys. 2021) → `sources/polymer_structure.pdf`
- **Sharma**, *Model-free analysis of SAXS/SANS: methodologies and pitfalls* (Soft Matter 2026) → `sources/model_free_analysis.pdf`

Master equation: `I(Q) = f·(Δρ)²·V²·P(Q)·S(Q) + B` — f volume fraction, Δρ
contrast, P(Q) form factor (shape), S(Q) inter-particle structure factor, B flat
(incoherent) background. **Dilute → S(Q)≈1**, so the job is mainly to identify P(Q).
Always work on dilute data first so S(Q) doesn't confound the form factor.

## 1. Read the curve first: shape → model

| log–log I(Q) shape | Physics | Model(s) |
|---|---|---|
| Low-Q plateau + knee + power-law decay | finite particle; plateau→size, knee at Q≈1/Rg | `guinier_porod`, `unified_power_Rg` (Beaucage), or a specific form factor |
| Pure power law (no plateau in window) | object outside window; fractal/interface only | `power_law`/`porod`; read exponent (§3). Guinier is INVALID here |
| Plateau → Q⁻² (Kratky plateau at high Q) | Gaussian/ideal chain (θ-solvent, melt, ν=½) | `mono_gauss_coil` (Debye), `poly_gauss_coil` if polydisperse |
| Plateau → Q^(−5/3) | swollen chain, good solvent (ν≈0.6) | `polymer_excl_volume` |
| Plateau + Lorentzian roll-off, no sharp knee | semidilute solution / gel; mesh size ξ | `correlation_length`, `lorentz` (OZ), `gauss_lorentz_gel` |
| Low-Q power-law tail + Lorentzian | network + clusters/aggregates + mesh | `correlation_length`: `A/Q^n + C/(1+(Qξ)^D) + B` |
| Peak (ring) | inter-particle/domain correlation; microphase sep.; d=2π/Q_peak | `broad_peak`, block-copolymer, or a lattice S(Q) |
| Oscillations / sharp dips | monodisperse spheres, smooth interface | `sphere`, `core_shell_sphere` (R from first dip) |
| Two power laws (e.g. Q⁻¹ then Q⁻⁴) | anisotropic, two length scales (rod length, then cross-section) | `cylinder`, `flexible_cylinder` |
| High-Q Q⁻⁴ | sharp two-phase interface | `porod` (deviations → surface fractal / diffuse interface) |
| Steep low-Q UPTURN (Q⁻³…Q⁻⁴ rising toward Q→0) | aggregation / large-scale structure — usually OUT OF SCOPE | EXCLUDE it (set q_min); do not force the form-factor model to fit it |

## 2. Polymer / soft-matter models (form, params, regime)

- **`mono_gauss_coil`** (Debye): `P=(2/x²)(e^{−x}+x−1)`, x=Q²Rg². Ideal chains only
  (ν=½): θ-solvent or melt. High-Q → Q⁻². `poly_gauss_coil` adds polydispersity.
  Do NOT use for swollen/collapsed chains. [Wei-Hore, Hammouda]
- **`polymer_excl_volume`**: swollen chain, fits Rg and Flory exponent ν (valid
  ν∈0.3–0.8; good solvent ≈0.6). High-Q → Q^(−1/ν); fractal dim D=1/ν. Use a thin
  rod instead as ν→1. [Wei-Hore]
- **`correlation_length`** (Ornstein–Zernike + Porod):
  `I = porod_scale/Q^porod_exp + lorentz_scale/(1+(Qξ)^lorentz_exp) + background`.
  Two physical terms: the **Porod term** (`porod_scale/Q^porod_exp`) captures a
  **low-Q UPTURN / large-scale structure**, and the **Lorentzian term**
  (`lorentz_scale`, `cor_length` ξ) captures the **correlation/mesh** feature. The
  right choice for semidilute solutions, gels, networks. **How to fit it (general):
  FIX `lorentz_exp = 2` (standard OZ) and FIT `porod_scale, porod_exp,
  lorentz_scale, cor_length, background` together.** This model CAN fit a low-Q
  upturn — it is just a matter of balancing `lorentz_scale` vs `porod_scale` (the
  Porod term absorbs the upturn, the Lorentzian the mid-Q roll-off). So you usually
  do NOT need to exclude the upturn for this model — fit a wide Q-range. If there
  is no upturn, `porod_scale` simply fits to ~0. [Wei-Hore, Hammouda]
- **`cylinder` / `flexible_cylinder`**: rods, fibers, semiflexible chains,
  bottlebrush backbones. Q⁻¹ (length) then Q⁻⁴ (cross-section). Rg²=L²/12+R²/2. [Hammouda, Wei-Hore]
- **`sphere` / `core_shell_sphere` / `ellipsoid`**: compact particles, micelles,
  grafted NPs; add polydispersity to wash out form-factor minima. [Wei-Hore]
- **`star_polymer`**: f arms, Rg(arm), ν; Kratky develops a peak as arm number
  rises (mass near core). [Wei-Hore]
- **`rpa`**: polymer blends/solutions near a phase boundary; fits χ (Flory–Huggins)
  + component Rg. I(0)∝(T−Tc)⁻¹ (LCST: I(0) rises on heating; UCST: falls). [Wei-Hore, Hammouda]
- **Fractals**: `mass_fractal` (D=1–3), `surface_fractal` (rough interface). Beaucage
  `unified_power_Rg` for multi-level / hierarchical structures. [Sharma, Wei-Hore]

## 3. Power-law exponent → structure (slope of log–log I vs Q)

| I∝Q^(−m) | structure |
|---|---|
| −1 | rigid rod / thin cylinder |
| −5/3 (≈−1.67) | swollen chain, good solvent (ν=0.6) |
| −2 | Gaussian/ideal chain (ν=½) — also any flat 2D sheet (not unique!) |
| ~−3 | collapsed/globular (ν=⅓) or compact/branched |
| 1<m<3 | mass fractal, D=m |
| 3<m<4 | surface fractal, I∝Q^(−(6−Ds)), 2<Ds<3 |
| −4 | smooth sharp interface (Porod) |
| >4 with roll-off | diffuse interface: I=Kp·Q⁻⁴·exp(−Q²a²) |

An exponent is necessary but NOT sufficient — Q⁻² is an ideal coil OR a flat sheet.
Corroborate with the rest of the curve and the known chemistry. [Wei-Hore]

## 4. Model-free analyses (Q-range & pitfalls)

- **Guinier** `I=I₀·exp(−Q²Rg²/3)` (ln I vs Q²): valid **Q·Rg < 1.3** (≈<1.0 if
  anisotropic). Iterate: fit low-Q, restrict to Q<1/Rg, recompute until Rg
  converges. Requires dilute, monodisperse, S(Q)≈1. **Use Guinier ONLY when a real
  low-Q plateau/knee exists** — never on a pure power law or a curve still rising
  at low Q (Rg becomes meaningless, often pinned at a bound). Polydispersity biases
  Rg HIGH (intensity ∝ V²). Low-Q upturn=aggregation; downturn=repulsive S(Q). [Sharma, Wei-Hore]
- **Porod** high-Q Q⁻⁴: detect via an **I·Q⁴ vs Q plateau** = Kp → surface area.
  Positive deviation (I·Q⁴ rising) = additive (bad background) — fix background
  first. Negative deviation = diffuse interface `Kp·Q⁻⁴·exp(−Q²a²)`. [Sharma, Hammouda]
- **Kratky** Q²I vs Q: plateau = Gaussian chain; upward = swollen; peak = compact.
  (Use Q·I vs Q for rods.) [Wei-Hore, Hammouda]
- **Guinier–Porod** `guinier_porod`: bridges a Guinier knee to a power law; shape
  factor **s=0 sphere, s=1 rod, s=2 plate** (non-integer allowed). Good default when
  one model must span a knee + slope. [Wei-Hore]
- **Invariant** ∫I·Q²dQ → volume fraction (needs absolute I + extrapolation). [Sharma]

## 5. Fitting strategy & pitfalls (apply by default)

- **Initial guesses from the data**: Rg from the Guinier slope; I₀ from its
  intercept; ξ from the Lorentzian roll-off (ξ≈1/Q_roll); shape/fractal exponent
  from the high-Q slope; d=2π/Q_peak; sphere R from the first dip. Good starts
  matter — a far-off start traps local optimizers (use a global search then refine).
- **Fix what the data can't constrain**: if MW, core size, or grafting density are
  known, fix them so P(Q) depends mainly on size + shape; let scale/background and
  the key shape parameters float. Don't float two coupled prefactors at once.
- **Choose the Q-window deliberately**: exclude the low-Q aggregation upturn with
  q_min (but never cut into the knee that sets the size). The 1-2 LOWEST-Q points
  are frequently beam-stop/mask artifacts — exclude them. A model valid over a
  LIMITED Q-range is still informative — state its range of validity.
- **Keep the high-Q plateau when you fit a background**: the flat high-Q level IS
  the incoherent background. If `background` is a fitted parameter, do NOT cut
  q_max short of that plateau (extend to ~0.4 A^-1 / the data end) or the
  background — and hence the whole fit — will be poorly constrained.
- **Background**: incoherent (H-rich) scattering is a flat B that must be MEASURED
  (blank/empty cell), not computed. Over/under-subtraction distorts the high-Q
  exponent → false fractal/Porod conclusions. Inspect subtracted data on a LINEAR
  scale (offsets are invisible on log–log); drop zero/negative points.
- **Judge by eye, not χ² alone**: χ²_R→1 ideal, χ²_R≪1 = overfit, and a visually
  good fit can be physically wrong. Check residuals (random vs systematic) and
  **validate parameters against physical constraints** (densities, sizes, volume
  conservation). A parameter pinned at a bound = wrong model/window/start. [Wei-Hore, Sharma]
- **Robustness**: confirm fitted parameters are stable against fit-range and
  background-level changes. Watch for polydispersity (washes out minima, shifts
  Porod), multiple scattering (keep transmission >60%), and resolution smearing
  (reduces apparent Rg, softens dips).

## 6. Practical fitting workflow (use this order, every time)

Hard-won lessons that apply to ANY experiment — follow this sequence:

1. **Match the incoherent background FIRST.** The flat high-Q level IS the
   incoherent background. Read it off the high-Q plateau (≈ the median of the
   highest ~20% in Q) and use it as the INITIAL background before fitting.
   Ornstein–Zernike / `correlation_length` and Lorentzian fits are extremely
   sensitive to the background start — getting it wrong throws the whole curve
   off. If the model line sits visibly above/below the high-Q data, the
   background is wrong; fix it and refit.
2. **Set the fit window deliberately — the low-Q decision is MODEL-dependent.**
   Always drop the 1–2 lowest-Q beam-stop/mask artifact points, and always KEEP the
   high-Q plateau (extend q_max to the data end, ~0.4 Å⁻¹) when `background` is free.
   Then, for the low-Q rise:
   - If the model has a low-Q power-law / Porod TERM (e.g. `correlation_length` has
     `porod_scale/Q^porod_exp`; Beaucage/unified models), DON'T exclude the upturn —
     fit the FULL Q-range and let that term absorb the upturn while the other term
     (Lorentzian/Guinier) captures the feature. This reliably fits low-Q-upturn
     curves; it's just a matter of balancing the two amplitudes.
   - If the model has only ONE feature term (single-chain `mono_gauss_coil`,
     Guinier, a sphere/ellipsoid form factor), it CANNOT represent an extra low-Q
     upturn — exclude the aggregation upturn with q_min (without cutting the knee).
3. **Seed the other parameters from data features.** Rg or ξ ≈ 1/Q_knee; scale
   from the low-Q level; shape/fractal exponent from the high-Q slope. Fix what
   the data can't constrain.
4. **Fit, then look — judge by eye.** If the model line is clearly off the data
   ANYWHERE inside its claimed window, the fit is bad regardless of χ². χ²_R→1 is
   ideal, χ²_R≪1 is overfit. A parameter pinned at a bound = wrong
   model/window/start. Residuals must be random, not systematic.
5. **Escape local minima.** A far-off start traps local optimizers; if the local
   fit is poor or a parameter hits a bound, run a global search (differential
   evolution) then refine locally.
6. **Refine once if warranted.** If residuals are systematic at low or high Q,
   adjust the window (drop a few more low-Q points, or extend high-Q for the
   background) and refit one more time; keep it only if it genuinely improves.
7. **Always extrapolate the model beyond the fit window (dashed) when plotting.**
   It shows where and how the model breaks down outside its range — e.g. a
   single-chain/OZ model under-predicting a low-Q aggregation upturn is exactly
   the diagnostic a reader needs.
8. **A limited-range fit is valid and informative** — state its range of validity
   and attribute the out-of-range deviation (aggregation, large-scale structure,
   background) rather than forcing one model across the whole curve.

When in doubt about whether to fit at all: data + a careful QUALITATIVE
description (shape, trends across the series, plausible interpretations — a slope
has several possible meanings) is often more honest than a forced model fit.

## 7. Q ↔ length-scale intuition

Small Q → large length scales, large Q → small. Periodic spacing d≈2π/Q_peak;
diffuse correlation/size ≈1/Q (Guinier knee at Q≈1/Rg). Decade map: below 1/Rg =
whole-particle plateau / aggregates; ~1/Rg = Guinier knee (overall size);
intermediate = chain-conformation / mass-fractal power law; highest Q =
surface/interface Porod. The required Q-range is NOT universal — the feature of
interest must lie inside the measured window or the analysis fails.

## 8. The Guinier-region prerequisite (decisive — obey the curve, not the proposal)

Every model that reports a SIZE — `guinier`, `guinier_porod`, `mono_gauss_coil`,
`poly_gauss_coil`, `polymer_excl_volume`, `unified_power_Rg`, and all form factors
(`sphere`, `ellipsoid`, `cylinder`, …) — requires a **resolved Guinier region**: a
low-Q PLATEAU where I(Q) flattens (log–log slope → 0) before the higher-Q power
law. That plateau is what pins Rg.

- If the low-Q keeps RISING (an upturn, or an unbroken power law with no
  flattening), the size is **unconstrained**. Such a model fails at low Q — it
  rolls over while the data shoots up — and the reported Rg is meaningless.
- In that case use a **correlation / Ornstein–Zernike** description
  (`correlation_length`, `lorentz`): it needs no Guinier plateau and fits the full
  range. For any solution / semidilute sample, always *compare* these two.
- **Excluding the low-Q with q_min to prop up a size model is not acceptable.** A
  correlation model that fits the whole range beats a size model that only "works"
  on a truncated window.
- `guinier_porod` is for a genuine TWO-LEVEL system (a globular Guinier knee that
  IS in range, plus separate large-scale structure) — not for absorbing an upturn.
- Caution: a Lorentzian/OZ curve also flattens at low Q. A plateau therefore makes
  size models *admissible*, not correct; absence of one is the strong signal.

## 9. Prefer the full measured Q-range

Default q_min/q_max to the data ends. Narrow the window ONLY for a genuinely
out-of-scope feature (an aggregation upturn the model cannot describe, a
noise-dominated tail), never to lower χ².

- After choosing a model on a restricted window, **refit it over the full range**
  and keep the full-range fit unless it is clearly worse. If the model's
  extrapolation already tracks the excluded points, excluding them was cosmetic.
- A fit that excludes points and then **diverges from them** is worse than it
  looks: judge the model against the data it threw away, not only the data it kept.
- `correlation_length` has a Porod term precisely so it can describe a low-Q
  upturn directly — with it, a restricted window is rarely justified.

## 10. Searching parameter space, and judging the result

**Search — one start is not enough.** A single local optimisation from one seed
lands in a local minimum for `correlation_length`, `lorentz`, `teubner_strey`,
`broad_peak` and friends. Fit each candidate from MANY diverse starts: sweep the
size parameter around the data's knee (roughly 0.15×–6×) and the power-law
exponents across their physical range; keep whichever converges best. Anchor
amplitude/scale/background to the measured intensity rather than randomising them.

**Judge by curve shape, not χ² alone.** χ²_R is a supporting witness. Rank fits on
how well they reproduce the *shape*:
- the size of the log–log residual (the vertical gap a human sees);
- agreement of the LOCAL SLOPE profile across the whole range (curvature everywhere);
- agreement of FEATURE POSITIONS (knee, bump, valley in the right place);
- residual RANDOMNESS — long same-sign runs mean systematic misfit even at low χ².

A low-χ² fit with an S-shaped residual, or a knee in the wrong place, loses to one
that tracks the curve everywhere.

**Then apply physics penalties** to that score, because curve agreement alone
cannot see:
- a size-based model used where there is no Guinier plateau (§8);
- points excluded from the fit that the model then diverges from (§9);
- a parameter pinned at a bound, or with relative uncertainty ≳1 — that parameter
  is unconstrained and the model/window/start is wrong.

## 11. Peaks and repeat distances

When the science question is a PERIODICITY — lamellar repeat, interlayer or layer
spacing, granum/thylakoid stacking, any "d-spacing" — the experiment is answered by
one number per sample: `d = 2π/Q_peak`. Report it in Å *and* nm; the literature for
biological and lamellar systems is usually in nm.

Finding the peak, in order of reliability:

1. **Fit a peak-bearing model** (`broad_peak`, `gaussian_peak`, `lamellar`,
   `teubner_strey`) and take its peak parameter (`peak_pos`, `q0`, `d_spacing`).
   This is the best estimate: the fit separates the peak from the decaying
   background and gives an uncertainty (propagate it: σ_d = 2π·σ_Q/Q²).
   Never choose a Q-window that cuts the peak out.
2. **Model-free**, as a cross-check and for curves that are not fitted. A
   correlation peak usually rides on a steeply decaying background, so the raw
   curve has NO local maximum where the peak is. Remove a smooth baseline in
   log–log first (a low-order polynomial; a straight line is not enough — the
   curvature at the window ends then dominates and the search lands on an edge),
   then find the maximum of the residual.

Guard rails, learned the hard way:
- Require an **interior** maximum that stands several times above the
  point-to-point scatter. Otherwise noise is reported as structure.
- **Do not search the background-dominated high-Q tail.** A "peak" at Q≈0.4 Å⁻¹
  is a 1.5 nm "repeat distance" — confident nonsense. Confine the search to the
  lower portion of the measured range unless the science says otherwise.
- Report "no peak found" honestly. A curve without a clear peak is a result.
- Model-free and fitted estimates can disagree (the fit sees through the
  background, the model-free estimate is pulled by it). **Report both.** The
  disagreement measures how well separated the peak is, and is information.
- Track the peak ACROSS the series — how d changes with temperature, position or
  treatment is usually the actual science, not any single value.

### 11a. Count the peaks before choosing a peak model

A stacked or lamellar system rarely gives ONE peak: expect a dominant order plus
weaker ones, sometimes finer than they first appear. This matters because a single
broad component (`broad_peak`) will fit one wide bump straight across several
finer peaks — the fit looks plausible and the position and width it reports belong
to no actual peak.

- **Count first, model second.** Locate every peak that clears the noise, then
  choose a description that can represent that many.
- For several peaks, an **empirical** description is more honest than forcing one
  physical model: a correlation-type background (Porod + Lorentzian + flat) plus
  ONE GAUSSIAN PER PEAK. Each peak then has its own position and width, each with
  an uncertainty.
- **Report the WIDTH as well as the position.** The position gives the repeat
  distance; the width says how ordered it is — a broad peak means a widely
  distributed spacing, and a change in width across a series is a real result
  independent of any change in d.
- Choose the number of peaks by an information criterion (BIC), not by adding
  components until chi-squared stops falling, and reject components that are
  really background: much wider than a real peak, centre not located to better
  than its own width, or sitting on top of a stronger peak.
- **Once the peaks are measured this way, a further single-model fit adds little.**
  Judge any model on whether it reproduces the peak STRUCTURE; if none does, say so
  rather than quoting the parameters of a model that smoothed the structure away.

## 12. Planning the analysis (before any fitting)

Deciding WHAT to analyse and HOW TO GROUP it determines whether a report is
useful. These rules apply to any experiment.

**Which reduced data to use**

- **Prefer combined / stitched / merged extended-Q profiles** over
  per-configuration curves whenever they exist. A merged profile joins the
  detector configurations into one curve spanning the full Q-range; fitting or
  plotting a single configuration throws away the length scales that the other
  configuration measured, and both the plots and the fits are markedly worse.
  Names vary by experiment (`merged_*`, `*_stitched`, `*_combined`) — check what
  is actually present rather than assuming.
- If several reduction variants exist (different masks, different reduction
  runs), pick ONE deliberately and say why. **Never compare a variant that has
  merged data against one that does not** — the difference you see will be the
  reduction, not the sample.
- The 1–2 lowest-Q points are frequently beam-stop or mask artefacts. Exclude
  them; do not let them drive model choice.

**What counts as science data**

- Exclude calibration standards and instrument references (porasil/porsil, blank,
  empty cell) from the science groups — they are not samples.
- Exclude solvent and background measurements (D2O, H2O, buffer, empty banjo)
  from sample comparisons, while remembering that the incoherent background they
  define is what makes the high-Q level meaningful.

**Grouping**

- Group by the INDEPENDENT VARIABLE the experiment actually varied: temperature,
  concentration, salinity, position, treatment. The comparison a reader wants is
  one curve per condition on a single plot.
- **Every science sample must appear in some group.** A grouping that covers only
  part of the measured samples silently hides data; if a proposed grouping leaves
  most samples out, fall back to a simple deterministic grouping (one bucket per
  sample stem and condition) rather than reporting a convenient subset.
- Keep genuinely different things apart: different sample thickness, different
  cell, or a different reduction are not members of one series.
- Order series members by their numeric condition (temperature, concentration),
  not alphabetically, so a trend reads correctly.

**Reporting**

- State what was NOT analysed and why. A caveat that names the missing data is
  worth more than a report that quietly omits it.
- Prefer a plain statement of what the data show over a model-derived number the
  data cannot support (see §8 and §10).

### 11b. Lamellar models: only the STACK models report a spacing

If the system is lamellar, choose a model that can actually answer the question:

- `lamellar_stack_caille`, `lamellar_hg_stack_caille`, `lamellar_stack_paracrystal`
  have a **`d_spacing` parameter** — these can report a repeat distance.
- plain `lamellar` and `lamellar_hg` are single-lamella **form factors** with NO
  spacing parameter. They describe one bilayer, not a stack, and cannot give a
  repeat distance no matter how well they fit.

**Always check the model's spacing against the measured peaks.** A model can win
on curve-shape agreement while reporting a spacing that corresponds to no observed
peak — the fit reproduces the overall decay and quietly ignores the peak
structure. If the fitted `d` does not match a measured peak position (or a
low-order reflection of one, d/2, d/3), the number is not the repeat distance and
must not be quoted as one; report the measured peak positions instead and say the
model failed to reproduce the structure.

And never report that a model "fails to capture the peak" without also giving the
measured peak position. A stated failure with no number is useless to the reader.

## 13. When NOT to fit a model at all

Model fitting is not the default form of analysis. Fit a model when the science
question IS a model parameter — and not otherwise.

- **Let the proposal decide.** If it says "determine Rg from the Guinier region",
  a Guinier analysis is exactly what the report should contain, and the proposal's
  model belongs first among the candidates. If the proposal asks for a repeat
  distance, the answer is the peak position — an empirical measurement that needs
  no model at all.
- **An unrequested model fit is worse than no fit.** It produces a confident
  number that answers nobody's question, and a model can win on curve-shape
  agreement while reproducing none of the structure that matters: one broad
  component smeared across several finer peaks, or a lamellar spacing
  corresponding to no observed peak. A wrong number in a report is more damaging
  than an absent one.
- **A default analysis is not a lesser one.** The curves, an honest qualitative
  description of what they show, and the measurement the proposal actually asked
  for (peak positions and widths, a trend across the series) answer the science
  question. Say what the data show; add a model only when it earns its place.
- When a model IS fitted and fails to reproduce the key feature, say so plainly
  and give the measured quantity instead. "The model does not describe the peak"
  plus the measured peak position is a result; the model's parameters alone are
  not.

### 13a. Presenting a series: grid the single-curve plots

A figure that shows ONE curve does not need the full page width. At that size the
reader gets one sample per figure and has to page back and forth to compare them,
which is the opposite of what a series report is for.

- Reserve full-width figures for plots that genuinely overlay SEVERAL datasets —
  a group overlay, a trend across a series.
- Put per-sample plots (individual fits, peak decompositions) into a **2- or
  3-column grid**, one panel per sample, on a single page. The comparison the
  reader actually wants — how the peak position, width or shape moves from sample
  to sample — then reads at a glance.
- Annotate each panel with its result (e.g. the repeat distance) so the grid is a
  summary in itself and not merely a contact sheet.
- Keep panels legible: at panel size the default log-axis ticks collide into an
  unreadable smear, so label only a few decades and drop the minor tick labels.

### 13b. Evidence belongs in the appendix, results up front

A figure whose purpose is to make a number CREDIBLE is evidence, not a headline.
Putting the peak-fit decompositions before the data plots makes the first thing a
reader sees a diagnostic, and buries the observation they opened the report for.

- Lead with the measurement: the curves, what they show, and the numbers.
- Put the supporting fits and decompositions in an **appendix**, and REFER to them
  from every place that quotes a number derived from them. A cross-reference costs
  the reader one page turn and buys full traceability.
- The rule generalises: anything included so a sceptical reader can check the work
  — fit decompositions, candidate-model comparisons, run catalogues — is appendix
  material. Anything that answers the science question is not.
