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
- **`correlation_length`** (Ornstein–Zernike + Porod): `A/Q^n + C/(1+(Qξ)^D) + B`
  → correlation/mesh length ξ. The right choice for **semidilute solutions, gels,
  networks** when a single-chain model misses the shape. Pure Lorentzian
  `C/(1+(Qξ)²)+B` (D=2) is the simplest version; `gauss_lorentz_gel` adds a second
  (static) length. [Wei-Hore, Hammouda]
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

## 6. Q ↔ length-scale intuition

Small Q → large length scales, large Q → small. Periodic spacing d≈2π/Q_peak;
diffuse correlation/size ≈1/Q (Guinier knee at Q≈1/Rg). Decade map: below 1/Rg =
whole-particle plateau / aggregates; ~1/Rg = Guinier knee (overall size);
intermediate = chain-conformation / mass-fractal power law; highest Q =
surface/interface Porod. The required Q-range is NOT universal — the feature of
interest must lie inside the measured window or the analysis fails.
