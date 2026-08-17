# Changelog

Versioning policy for `reportit`:

| Change | Bump |
|---|---|
| small change — bug fix, wording, tuning, a config/model swap | **+0.0.1** |
| major function addition — a new capability or CLI flag | **+0.1.0** |

The version lives in **one** place, `src/reportit/__init__.py`; `pyproject.toml`
reads it from there (hatch dynamic version) so the two cannot drift. It is shown
on every report's title page and page footer, at the top of
`sasfit_notebook.md`, in the CLI run output, and via `reportit --version`.

**When you change the code, bump `__version__` and add a line here.**

---

## 0.9.1

- **Knowledge guide gains section 12, "Planning the analysis".** An audit found
  the guide covered the fitting rules completely but carried NOTHING for the
  strategy stage — which has been receiving it since 0.8.0. Five rules lived only
  in `strategy/engine.py`'s prompt and in the grouping-guard code: prefer merged
  extended-Q profiles, never compare a merged variant against a non-merged one,
  exclude calibration standards and solvent/background runs, group by the
  independent variable with every science sample covered, and order series
  numerically. They are now stated generally, so the planning agent is guided by
  the same editable document as the fitter — and so a scientist can change how
  reportit plans an analysis without touching Python.

## 0.9.0

- Knowledge guide gains section 11a, "Count the peaks before choosing a peak
  model", so the lesson applies to future experiments and not just this code path.

- **Multi-peak analysis (+0.1.0), from operator feedback.** A stacked system shows
  SEVERAL finer peaks, and a lone `broad_peak` fit puts one wide component across
  all of them — reporting a position and width that belong to no actual peak.
  - `dspacing.find_peaks()` reports every peak clearing the noise, not just the
    strongest.
  - `dspacing.fit_peak_model()` fits the curve empirically as a correlation-type
    background (Porod + Lorentzian + flat) **plus one Gaussian per peak**, giving
    each peak its own position AND width with 1-sigma uncertainties. The number of
    peaks is chosen by BIC, so an extra peak must earn its parameters.
  - Fitted components that are background rather than peaks are rejected: too wide
    (> 1/3 of the fitted range), centre uncertain by more than its own FWHM or more
    than 20% of its position, or overlapping a stronger peak.
  - The per-group table now reports **peak index, Q, d (A and nm), and FWHM**, each
    with its uncertainty.
  - The model-selection prompt now warns that a single broad component across
    several finer peaks is a misfit, and notes that where peak positions and widths
    are already measured empirically a further single-model sasmodels fit adds
    little.
  - Validated on IPTS-38773: all three `leaf1_dark` curves resolve a reproducible
    SECOND peak at ~13.5 nm alongside the ~19 nm primary, which the previous
    single-peak treatment missed entirely; log-RMS residual improved
    (e.g. 0.0089 -> 0.0054).

## 0.8.1

- **Docs brought in line with the code.** The README named `glm-5.2` and "gemini"
  as the reasoning and vision models after those had been switched to
  `gemini-3.7-flash` — actively misleading, since it told the reader a model that
  is no longer used. Both now name the `.env` setting instead of a model, so they
  cannot go stale again.
- CLI options table completed: `--knowledge`, `--learn`, `--learn-stage`,
  `--learn-title`, `--show-knowledge`, `-V/--version`. Cross-checked against
  `--help`: every flag the CLI accepts is now documented.
- Pipeline diagram and the key-modules list updated with `guidance.py`,
  `shortnames.py`, `analysis/dspacing.py` and `analysis/knowledge.py`.

## 0.8.0

- **Teachable knowledge (+0.1.0).** reportit now gets smarter as you teach it.
  - `reportit --learn "..."` records a general lesson (optionally
    `--learn-stage`, `--learn-title`) into `~/.reportit/knowledge/`, read by every
    later run for every experiment. Lessons are grouped one file per stage set,
    since routing is a property of the file.
  - **Knowledge now reaches all four stages**, not just fitting: `strategy` (how
    to group samples and what to analyse), `fitting`, `critic`, `narrative`.
    Previously the strategy and narrative agents saw no reference knowledge at all,
    so lessons about analysis approach could never influence planning.
  - Notes may carry optional front matter (`title`, `applies_to`, `keywords`,
    `priority: always`) to say which stage they belong to. Files without front
    matter keep working unchanged as general guidance.
  - **Relevance selection**: while the library fits the per-stage budget everything
    is sent, exactly as before; once it outgrows it, notes are ranked against the
    actual experiment and `priority: always` notes are never dropped. Deterministic
    keyword matching — no embedding service, no new dependency.
  - **The knowledge is part of the LLM cache identity.** Without this a rerun would
    replay pre-lesson reasoning and teaching would appear to do nothing.
  - `--knowledge DIR` (repeatable) adds a library for one run; `--show-knowledge`
    lists what is loaded and which stage each note reaches.
  - The report records which notes each stage used, so an old report can be traced
    back to the knowledge that produced it.

## 0.7.1

- **Knowledge guide brought up to date with the rules the code had learned.**
  `knowledge/sans_analysis_strategy.md` had not changed since 2026-06-26, so the
  Guinier-region rule, the full-Q-range preference, multi-start search, shape-aware
  judging with physics penalties, and the peak / repeat-distance workflow existed
  only as Python prompt constants and algorithms. They are now written as general,
  cross-experiment guidance (sections 8-11) in the file that is loaded into the
  model-selector and fit-critic prompts on every run, for every experiment.

## 0.7.0

- **Repeat-distance / d-spacing analysis (+0.1.0).** New `analysis/dspacing.py`.
  When the proposal's goals or hypotheses are about a periodicity (repeat
  distance, d-spacing, lamellar, interlayer, granum/thylakoid, correlation or
  diffraction peak), the report now answers that question directly instead of
  leaving the reader to convert Q by hand:
  - every group gets a **Repeat distance table** — the correlation-peak position
    located *model-free* (a degree-4 log-log baseline is removed, then an interior
    maximum is required to stand 3x above the point-to-point scatter) and
    converted to `d = 2*pi/Q_peak`, in both A and nm, with the peak SNR;
  - the model-selection agent is told the experiment targets a repeat distance
    and to include a peak-bearing model, so the peak is actually *fitted*;
  - when the chosen model has a peak parameter (`broad_peak.peak_pos`,
    `gaussian_peak.q0`, `lamellar.d_spacing`, ...) the fitted repeat distance is
    reported with a propagated uncertainty in the fit section.
  Detection uses specific phrases only — bare "repeat"/"periodic"/"RD"/"stacking"
  and the polymer-science "repeat unit" are deliberately excluded, since under
  `--no-llm` the proposal summary can be raw PDF text and a loose match would fire
  on every experiment. Validated: fires on IPTS-38773 (granum repeat distance) in
  both LLM and `--no-llm` modes, and on neither mode for IPTS-37095.

## 0.6.1

- **Fix: short names now reach the model-parameter table and the fit figures.**
  The per-member fit table's `Member` column still carried the full output name,
  making the table overflow the page; it now uses the short name (fixed at the
  source in `sas_agent._fit_all_members`, so `sasfit_notebook.md` benefits too).
  The fit/trend figure captions and plot titles also used the raw group label.
- **The member table drops its `Condition` column when it duplicates `Member`.**
  For a group whose independent variable *is* the sample name the two columns were
  identical, costing width for nothing; a temperature series still shows both.
- Removed a dead `cond_val` assignment in `_fit_all_members`.

## 0.6.0

- **Short display names (+0.1.0).** New `shortnames.py` derives a readable label
  for long reduced-output names by dropping the leading and trailing name parts
  that every file in the experiment shares, so only the distinguishing part is
  shown (`merged_leaf1_370_70_4m_2p5A_30Hz_4m2.5a30hz_frame0_4m2.5a30hz_frame1`
  -> `leaf1_370_70`). Used in every table, figure legend, section title, caption
  and fit label; a **Short Name Legend** appendix table maps each back to its
  file. All-or-nothing per experiment, so a report never mixes derived names with
  opaque `S7` labels. Names <= 15 characters are never touched, and experiments
  whose names are merely long-ish (<= 30) are left alone.
- **Fix: Sample Summary drops columns it cannot fill.** Merged-only datasets have
  no reduction JSON, so Description / Configurations / Conditions were columns of
  dashes. Empty columns are now omitted, and if only the sample name survives the
  table is dropped entirely — the group sections and the ONCat catalog already
  carry that information.

## 0.5.4

- **Fix: reduced 1D data is found regardless of extension.** `*_Iq.txt` is now
  accepted alongside `*_Iq.dat` (`discovery/scan.py`, `discovery/inventory.py`) —
  the extension carries no meaning.
- **Fix: a folder holding only merged/combined profiles no longer scans as
  empty.** Merged files used to be indexed purely to be *attached* to a dataset
  created from a `*_Iq.dat` file, so a merge-and-publish directory with no `.dat`
  files produced zero datasets. A merged profile with no per-config counterpart
  now becomes a dataset in its own right. Found on
  `IPTS-38773/shared/cdo/output_merged` (0 -> 11 datasets); verified byte-identical
  dataset lists on five existing `.dat` experiments.

## 0.5.3

First version to record itself in its own output. The entries below account for
the work accumulated since 0.1.0 (the last tagged state), applying the policy
above: four major additions (+0.4.0) and three small ones (+0.0.3).

### Major additions (+0.1.0 each)

- **Autoresearch shape-aware fitting** — `sasresearch/` (`shape.py`, `search.py`)
  plus its integration into `analysis/sas_agent.py`. Every candidate model is fit
  from many starting points and ranked by a deterministic shape score
  (log-residual, slope profile, feature positions, residual runs test) adjusted by
  physics penalties; the LLM is a visual check, not the arbiter. Includes the
  Guinier-region rule and the full-Q-range preference. Standalone explorer:
  `python -m reportit.sasresearch <file>`.
- **Grouping guard** — `strategy/engine.py` replaces the LLM's grouping with a
  deterministic series grouping when its sample coverage falls below 80%, so a
  degenerate strategy run no longer leaves most samples ungrouped.
- **`--data DIR`** — name the reduced-data folder(s) explicitly instead of relying
  on auto-discovery. Repeatable, may point outside the target, and overrides both
  the LLM's variant choice and the merged-data guardrail. Also suppresses the
  variant deliberation in the strategy prompt, cutting agent steps and tokens.
- **`--userguide TEXT`** — steer the analysis in plain English. The instruction is
  interpreted once (`guidance.py`) and routed to the steps it affects: selection
  rules become a concrete dataset filter applied in code, advisory parts go only
  to the strategy / fitting / narrative prompts.

### Small changes (+0.0.1 each)

- **`--summary-only`** — build only `report_summary.pdf`.
- **All LLM roles switched to `google/gemini-3.7-flash`** (text+image,
  tool-calling, 1M context) — replaces gemini-3.5-flash and z-ai/glm-5.2.
- **Absolute output path fix** — `run_report` resolves `out_dir`, so a relative
  `-o` no longer breaks figure paths under `pdflatex`.

### Also in this release

- Model Selection & Fitting Rationale appendix + `sasfit_notebook.md` audit trail.
- Variant labels disambiguate directories that share a name (`<parent>/<name>`).
- Probe sandbox extended to the chosen data directories; strategy cache key now
  includes the data dirs, the fixed-dirs flag, and the user guidance.

## 0.1.0

Initial version: inventory → proposal → agentic strategy → execute → narrative →
LaTeX → `report_comprehensive.pdf` + `report_summary.pdf`.
