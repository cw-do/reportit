# SANS model-selection reference (sasmodels)

General guidance for choosing a SasView/sasmodels model and fitting window for
reduced 1D I(Q) small-angle neutron scattering data. This is reference knowledge,
not specific to any experiment. Add your own articles/notes alongside this file.

## First, read the curve shape (log-log I vs Q)
- **Low-Q plateau then a bend (knee), then power-law decay** → finite-size objects
  or correlation-length behaviour. Estimate a size from the knee: a feature at Q*
  implies a length ~ 1/Q* (Rg ≈ 1/Q_knee, correlation length ξ ≈ 1/Q*).
- **Featureless power law I ∝ Q^p across the whole range** → fractal / interfacial
  scattering; fit `power_law` or `porod` and report the exponent p.
- **A low-Q UPTURN that keeps rising toward Q→0** → almost always aggregation or
  large-scale structure *outside the length scale of interest*. Do NOT try to make
  the form-factor model fit it. EXCLUDE it with `q_min` just above the upturn and
  fit the regime the model applies to. The 1–2 lowest-Q points are also frequently
  beam-stop/mask artifacts.
- **A peak** → inter-particle correlations / structure factor (`broad_peak`,
  `peak_lorentz`, or a form×structure model).

## Model cheat-sheet for solution / soft-matter SANS
- **Single polymer chains (contrast-matched, dilute)**: `mono_gauss_coil` (Debye,
  ideal chain → Rg) or `poly_gauss_coil` (adds polydispersity). For swollen chains
  in good solvent use `polymer_excl_volume` (Rg + excluded-volume/Porod exponent).
- **Semidilute polymer solutions / gels / networks**: `correlation_length`
  (Ornstein–Zernike Lorentzian at low Q + Porod at high Q → correlation length ξ).
  This is often the right choice when a single-chain model misses the shape.
  `gauss_lorentz_gel` is an alternative for gels (static + dynamic correlation
  lengths).
- **Compact particles**: `sphere`, `core_shell_sphere`, `ellipsoid`, `cylinder`;
  add polydispersity if the form-factor minima are washed out.
- **Interfaces / sharp boundaries**: Porod slope −4 (`porod`); fractals →
  `mass_fractal` / `surface_fractal`.

## Guinier vs correlation length vs Porod
- Use **Guinier** (compact-particle Rg) ONLY when there is a clear low-Q plateau
  that bends into a knee. A curve that keeps rising at low Q, or a pure power law,
  is NOT a Guinier case — Guinier will give a meaningless Rg (often pinned at a
  bound). For many polymer-solution curves, **`correlation_length` or
  `polymer_excl_volume` works far better than Guinier.**
- If unsure between two models, propose BOTH (best-first) and compare.

## Choosing the fit window (q_min / q_max)
- Exclude the low-Q aggregation upturn with `q_min`, BUT do not cut into the
  knee/Guinier bend that constrains the size — put q_min just above the upturn.
- Use `q_max` to drop a flat, background-dominated high-Q tail.
- Reporting a model over a LIMITED Q-range is valid and informative; state the
  range of validity.

## Sensible initial guesses and parameter handling
- Estimate size parameters from the data (Rg or ξ ≈ 1/Q_knee), scale from the
  low-Q intensity, background from the high-Q plateau.
- Fit shape parameters + scale + background; fix what the data cannot constrain.
- A far-off starting value can trap a local optimizer — a global search
  (differential evolution) then local refinement is more robust.

## Judging the fit
- Reduced χ² and R² are necessary but NOT sufficient. Judge by eye: does the
  fitted curve follow the data across the whole fitted window? Are the residuals
  random, or systematically structured (model misses a regime)? A visually good
  fit with moderate χ² can be preferable to a low-χ² fit with systematic residuals.
- A parameter pinned at a bound is a red flag (wrong model, wrong window, or bad
  start), even if χ² looks acceptable.
