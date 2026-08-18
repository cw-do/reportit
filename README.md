# reportit

**LLM-driven post-experiment report generator for EQSANS** (Extended Q-range
Small-Angle Neutron Scattering) at the SNS, Oak Ridge National Laboratory.

Point it at an experiment's IPTS folder and it reads the proposal, figures out
what the experiment was about, inventories the reduced data, and writes a
generic-format LaTeX report (compiled to PDF) summarizing the science, showing
plots, and assessing whether the proposal's hypotheses are borne out by the data.

```bash
reportit 38533                 # -> ./reportit_out/report_*.pdf
reportit /SNS/EQSANS/IPTS-38533/shared -o /tmp/rep
```

See [Usage recipes](#usage-recipes-for-different-applications) for common
workflows (quick overview, full analysis, fitting-only exploration, reruns).

## What makes it different: the LLM drives the analysis

`reportit` does **not** hardcode the folder layout or analysis recipe. It hands an
organized inventory of the shared folder to an LLM, which then **iteratively
probes** the data with read-only tools — reading `NOTE.md`, listing datasets,
parsing reduction JSONs, looking up ONCat run titles, even sampling curve shapes
— until it understands the experiment. It then emits a structured
`AnalysisStrategy` deciding, per experiment:

- which reduced-output directory/variant is canonical (e.g. `output/` vs
  `output_mask4/`), and why;
- how to group datasets (temperature series, concentration series, config sets);
- whether 1D `I(Q)` overlays or 2D `I(Qx,Qy)` maps are the meaningful comparison;
- whether a quantitative model fit (Guinier `Rg`/`I0`, Porod/power-law slope) is
  scientifically sensible for each group.

The tool then executes that strategy, generates figures and fits, writes
narrative + a hypothesis assessment, and produces **two PDFs**: a
`report_comprehensive.pdf` and a condensed `report_summary.pdf`.

## Pipeline

```
inventory ─▶ proposal (pypdf + LLM) ─▶ user guidance (routed) ─▶ STRATEGY (agentic LLM + probes)
          ─▶ execute (load, metrics, peaks/d-spacing, plots) ─▶ sasmodels fitting
          ─▶ narrative (LLM) ─▶ assemble LaTeX ─▶ pdflatex ×2
          ─▶ report_{comprehensive,summary}.pdf + sasfit_notebook.md

Reference KNOWLEDGE (teachable, see below) feeds the strategy, fitting,
critic and narrative steps on every run.
```

Key modules (`src/reportit/`): `discovery/` (folder inventory, name parsing,
reduction-JSON), `integrations/oncat.py` (run catalog via pyoncat),
`proposal/` (PDF text + LLM summary), `llm/` (OpenRouter client with
caching, JSON, and the tool-calling loop; probe tool specs), `strategy/`
(the agentic engine + read-only probes), `analysis/` (native numpy loaders,
metrics, scipy fits — **no drtsans dependency**), `plotting/figures.py`,
`execute/runner.py`, `narrative/synthesize.py`, `report/` (jinja2 templates +
pdflatex), `guidance.py` (routes `--userguide` to the stages it affects),
`shortnames.py` (short display labels for long filenames),
`analysis/dspacing.py` (peaks → repeat distance), `analysis/knowledge.py`
(the teachable reference-knowledge library). Every ONCat / LLM / probe result is cached under
`<out>/.reportit_cache/`, so reruns are fast and deterministic.

## Install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e .            # uses the ORNL repoman index for pyoncat
```

Requires a system `pdflatex` (e.g. TeX Live) to produce PDFs; without it the tool
still writes the `.tex` files.

## Configuration

Copy `.env.example` to `.env` and set the OpenRouter key (used for all LLM steps):

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemini-3.7-flash            # strategy + narrative
OPENROUTER_FALLBACK_MODEL=google/gemini-3.7-flash   # retried if the primary call fails
OPENROUTER_REASONING_MODEL=google/gemini-3.7-flash  # model selection / fit critique
OPENROUTER_VISION_MODEL=google/gemini-3.7-flash     # visually inspect fit plots
```

All four roles default to `google/gemini-3.7-flash` (text+image, tool-calling,
1M context). Point any of them at a different model to specialise a stage — e.g.
a dedicated reasoning model for fit selection and critique.

> Changing a model does **not** invalidate the caches: most cache keys are
> semantic (`strategy:…`, `sasselect:…`) and carry no model name, so a rerun in
> an existing output dir replays the *previous* model's answers. Add `--refresh`
> on the first run after a model change.

The `.env` is gitignored — never commit it.

## Model-based fitting (opt-in; `--sasfit` to enable)

Model fitting is **off by default**. A model fit is only worth doing when the
science question *is* a model parameter; an unrequested fit produces confident
numbers that answer nobody's question, and a model can win on curve-shape
agreement while reproducing none of the structure that matters.

It runs when you ask (`--sasfit`), or automatically when **the proposal names the
analysis it wants** — "determine Rg from the Guinier region" enables a Guinier
analysis and puts it first among the candidates. `--no-sasfit` forces it off.

A default run still gives you the curves, the qualitative observations, and the
empirical peak / repeat-distance analysis with its plots — which is the complete
answer when the proposal's goal is a spacing rather than a model parameter.

When it does run, it fits each sample group the way a human analyst would: try several physically-plausible models,
fit each from many starting points to escape local minima, and rank them by a
**shape-aware score** that judges the fit both quantitatively *and*
qualitatively. Per group:

1. **Observe** — characterise the curve shape (local slope profile, the knee,
   any bump/valley, plateaus).
2. **Hypothesise** — a **reasoning agent** (`OPENROUTER_REASONING_MODEL`) proposes candidate
   SasView/sasmodels models + parameter plans (initial guesses, fit-vs-fix,
   bounds, optional Q-window) from the shape + proposal context + reference
   knowledge. (Falls back to a soft-matter shortlist if no LLM.)
3. **Fit robustly** — each candidate is fit with **bumps** from *many* diverse
   starting points (size swept around the knee, exponents over their range),
   with a global-search (differential-evolution) backstop. This multi-start is
   what fixes fits that a single least-squares start gets wrong.
4. **Judge (deterministic)** — every candidate is ranked by the **shape score**:
   log-residual size, slope/curvature agreement, knee/bump/valley position
   agreement, and residual randomness (a runs test). The score — *not* reduced
   χ² alone — chooses the model, so a low-χ² fit with a systematic residual or a
   knee in the wrong place loses to one that tracks the curve everywhere.
5. **Check (LLM)** — a multimodal critic (`OPENROUTER_VISION_MODEL`) then visually inspects the
   winning fit and flags any physical problems (accept / reject + assessment).
6. **Trend** — every member of the group is fit with the chosen model; the
   report tabulates and plots the key parameter across the series (e.g.
   correlation length or Rg vs temperature).

Partial-Q-range fits are first-class: a low-Q upturn (aggregation / large-scale
structure outside the length scale of interest) can be excluded so the model is
fit only over the regime it applies to; excluded points are shown faintly.

**Where the reasoning shows up.** The comprehensive report's *Model-Based
Fitting* section shows the chosen model, parameters, fit figure, per-member
trend, and the critic verdict (**including honest failures**). An appendix,
**"Model Selection & Fitting Rationale,"** records for each group the full
candidate table (every model tried, its shape score, χ², critic verdict), a
plain-language *why this model won*, the selector's rationale, and the critic's
assessment. The same audit trail is also written as **`sasfit_notebook.md`**
next to the PDFs.

### Standalone fitting explorer

To develop or debug the fitting on a single curve without running a whole
report, use the standalone explorer:

```bash
# fit one I(Q) file; writes plots + a notebook + results.json under ./sasresearch_out/
python -m reportit.sasresearch <path/to/..._Iq.txt|_Iq.dat>

python -m reportit.sasresearch DATA_Iq.txt --no-llm            # engine only, no API calls
python -m reportit.sasresearch DATA_Iq.txt --models correlation_length,lorentz,teubner_strey
python -m reportit.sasresearch DATA_Iq.txt --context "polymer in D2O, dilute" --n-starts 12
```

Outputs (in `./sasresearch_out/<stem>/`): `best_fit.png`, per-candidate
`candidates/*.png`, `results.json` (every model + every start, params ± errors,
sub-scores), and `notebook.md` (the ranked lab notebook). Options: `--out DIR`,
`--models a,b,c`, `--context TEXT`, `--n-starts N` (default 8), `--no-llm`,
`--no-refine`, `--refresh`, `-v`.

## CLI options

| Flag | Effect |
|------|--------|
| `-o, --out DIR` | output directory (default `./reportit_out`) |
| `--strategy-only` | print the LLM-derived `AnalysisStrategy` JSON and stop |
| `--no-llm` | deterministic mode: heuristic grouping, no LLM reasoning |
| `--proposal PATH` | proposal folder or single PDF to use (default: auto-discover `<shared>/proposal/*.pdf` and any `*.pdf` under `<shared>`; if none on disk, downloaded from ONCat by IPTS) |
| `--no-proposal` | ignore the proposal PDF(s) |
| `--data DIR` | reduced-data folder to analyze (default: auto-discovered — every directory under the target that holds `*_Iq.dat`). Repeatable; may point outside the target; an explicit choice overrides the LLM's variant selection |
| `--userguide TEXT` | a few sentences in plain English steering the analysis; interpreted once and routed to the steps it affects (alias `--guide`) |
| `--refresh` | bust caches (re-query ONCat / re-run LLM) |
| `--sasfit` / `--no-sasfit` | force model-based fitting on / off. **Off by default** — with neither flag it runs only when the proposal names the analysis it wants. Alias: `--sasmodels` / `--no-sasmodels` |
| `--summary-only` | generate only `report_summary.pdf` (skip the comprehensive report) |
| `--knowledge DIR` | extra directory of reference-knowledge notes for this run (repeatable) |
| `--learn TEXT` | teach a general lesson (written to `~/.reportit/knowledge/`) and exit |
| `--learn-stage S` | which stage(s) a `--learn` lesson applies to: `strategy,fitting,critic,narrative,all` |
| `--learn-title T` | short title for a `--learn` lesson |
| `--show-knowledge` | list the knowledge notes in use, per stage, and exit |
| `--max-llm-steps N` | cap on agentic strategy tool-calling steps (default 40) |
| `-V, --version` | print the reportit version and exit |
| `-v, --verbose` | verbose logging (shows each strategy probe) |

## Usage recipes for different applications

**Default run** — proposal + strategy + figures + qualitative observations +
narrative, plus any analysis the proposal's goal implies (peak / repeat-distance
measurement with its evidence figure). Model fitting runs only if the proposal
names the analysis it wants:

```bash
reportit 38533
```

**Force the model-based fitting on** — a full sasmodels search per group, adding
the Model-Based Fitting section and `sasfit_notebook.md`:

```bash
reportit 38533 --sasfit
```

**Quickest readable report** — condensed summary PDF only, fitting explicitly off:

```bash
reportit 38533 --no-sasfit --summary-only
```

**Everything, but only the summary PDF** — skip building the long comprehensive
report:

```bash
reportit 38533 --sasfit --summary-only
```

**Deterministic / offline (no API key, no LLM)** — heuristic grouping, default
model shortlist, no vision critic; useful for a sanity pass or when OpenRouter
is unavailable:

```bash
reportit 38533 --no-llm
```

**Inspect the plan without building anything** — print the LLM's
`AnalysisStrategy` (grouping, variant choice, fit decisions) and stop:

```bash
reportit 38533 --strategy-only
```

**Point at a specific folder / proposal, custom output location:**

```bash
reportit /SNS/EQSANS/IPTS-38533/shared -o /tmp/rep --proposal /path/to/proposal.pdf
```

**Name the reduced-data folder explicitly** — when auto-discovery picks up more
directories than you want (autoreduce output, an old reduction), or the reduced
data lives somewhere other than the experiment folder:

```bash
reportit /my/experiment --data /my/experiment/output_v3
reportit /my/experiment --data /scratch/reduced/run_final     # outside the target
```

The proposal is still auto-discovered from the target folder, so this composes
with `--proposal`. Repeat `--data` to analyze several reductions as comparable
variants (directories sharing a name are labelled `<parent>/<name>`):

```bash
reportit /my/experiment --data .../output --data .../output_mask4
```

An explicit `--data` is final: it overrides both the LLM's variant choice and
the merged-data guardrail, so you get exactly the folders you named. It also
makes the run **cheaper**, because no other data folder is considered at all:
only the named folders are scanned, and the strategy agent is told the variant
question is settled instead of being asked to investigate and decide it. On a
5-output-dir experiment (IPTS-38603) that cut the datasets scanned from 39 to
16, the `list_datasets` tool response from ~2100 to ~800 tokens (and that
response is re-sent on every subsequent agent step), and the agentic loop from
28 steps to 22 — reaching the same strategy.

**Fresh run (ignore caches)** — re-query ONCat and re-run every LLM step:

```bash
reportit 38533 --refresh
```

**Fitting-only exploration on one curve** — no report; iterate on models/starts
(see [Standalone fitting explorer](#standalone-fitting-explorer)):

```bash
python -m reportit.sasresearch merged_..._Iq.txt --models correlation_length,lorentz
```

> The output directory may be relative or absolute (`-o reportit_new` works); it
> is resolved to an absolute path internally so `pdflatex` always finds the
> figures. Every ONCat/LLM/fit result is cached under `<out>/.reportit_cache/`,
> so a second run of the same experiment is fast — add `--refresh` to bust it.

## Steering the analysis in plain English (`--userguide`)

Tell the tool what you want in a sentence or two, instead of hunting for a flag:

```bash
reportit /my/experiment --data ./usethis \
  --userguide "use merged files only for the analysis and summary"

reportit 38533 --userguide "skip the porsil standard and any banjo background"
reportit 38533 --userguide "group by salinity, not temperature; prefer correlation length"
```

The instruction is **interpreted once** and then **routed to the steps it
actually affects** — it is not pasted into every prompt:

| Part of the instruction | What happens |
|---|---|
| a data-selection rule ("merged only", "skip porsil", "only the 30C runs") | becomes a concrete filter applied **in code**; the narrowed dataset list is what every later step receives |
| a grouping / comparison hint | appended to the strategy prompt only |
| a model / fitting hint | appended to the model-selection and fit-critic prompts only |
| a wording hint ("keep the summary short") | appended to the narrative prompts only |

So `--userguide "use merged files only …"` really does reduce the file list
before anything runs (e.g. 136 → 132 datasets), rather than asking the LLM to
remember the request at each stage. The routing is visible in the run log and
recorded in the report:

```
User guidance (--userguide): "use merged files only for the analysis and summary"
Interpreted as: Use only merged extended-Q data files for all analysis, fitting, and summary reporting.
--userguide data selection kept 132 of 136 datasets (...).
```

Safeguards: a rule that would discard **every** dataset is reported and *not*
applied (a misreading should not silently produce an empty report); patterns
that are not valid regular expressions are dropped with a warning; and the
instruction is mixed into the LLM cache keys, so changing it never replays the
previous run's answers. Under `--no-llm` the "merged only" case is still honoured
by a keyword reading, and anything else is passed through as free text.

## Graceful degradation

A missing proposal (none on disk and none in ONCat), a thin/image-only proposal, missing ONCat, or `--no-llm` never hard-fails: the
report falls back to a data-driven summary (heuristic grouping, ONCat/ filename
titles, templated observations) and records what was missing in a Caveats section.

## Teaching reportit (it gets smarter as you add knowledge)

reportit reads a library of **reference-knowledge notes** on every run and feeds
them to the agents that plan, fit, criticise and write. Add to that library and
every future experiment benefits — nothing is specific to one dataset.

Teach it a lesson in one command:

```bash
reportit --learn "Excluding low-Q to make a size model fit is never acceptable; \
prefer a correlation model that fits the full range." \
  --learn-title "Never trim low-Q to prop up a size model" \
  --learn-stage fitting,critic
```

Or drop files into a knowledge directory — Markdown, text, **or PDF** (a SANS
paper is text-extracted as-is):

| Location | Use |
|---|---|
| `--knowledge DIR` | a library for this run only (repeatable) |
| `$REPORTIT_KNOWLEDGE_DIR` | a shared/team library |
| `~/.reportit/knowledge/` | your personal notes — `--learn` writes here |
| `<repo>/knowledge/` | the curated guide shipped with reportit |

See what is loaded and where it goes:

```bash
reportit --show-knowledge
```

**Which stage a note reaches.** Optional front matter routes it; a plain
Markdown file with no front matter is general guidance for every stage:

```markdown
---
title: Lamellar systems need a peak model
applies_to: fitting, critic     # strategy | fitting | critic | narrative | all
keywords: lamellar, d-spacing, peak, stacking
priority: always                # never dropped when the library is large
---
Body of the lesson...
```

The four stages are **strategy** (how to read the folder, group samples, decide
what to analyse), **fitting** (which models to try), **critic** (whether a fit is
acceptable), and **narrative** (how the report is written).

**As the library grows** it is not blindly concatenated: while it fits the
per-stage budget everything is sent, and beyond that notes are ranked against the
actual experiment (proposal text, curve shape, candidate models) with
`priority: always` notes never dropped. The ranking is deterministic keyword
matching, so the same experiment always gets the same knowledge.

**Teaching actually changes the answers.** The knowledge is mixed into the LLM
cache keys, so a lesson invalidates exactly the stages it applies to — without
that, a rerun would replay the pre-lesson reasoning. Each report records which
notes each stage used, so an old report can be traced back to the knowledge that
produced it.

## Peaks and repeat distance / d-spacing

When the proposal is about a *periodicity* — a lamellar repeat, an interlayer
spacing, a granum thylakoid stacking distance — the experiment is answered by the
peak positions, converted to real-space repeats `d = 2*pi/Q_peak`.

reportit detects that from the proposal and then, for every group, reports a
**Peaks resolved** table: peak index, `Q_peak`, `d` in A and nm, and the peak
**FWHM**, each with its 1-sigma uncertainty.

**Several peaks, fitted as several peaks.** A stacked system usually shows more
than one peak — a dominant order plus weaker ones. Fitting a single broad
component across all of them is a misfit: the position and width it reports
belong to no actual peak. So the curve is described *empirically* as

```
I(Q) = [ Porod + Lorentzian + flat background ]  +  sum of Gaussian peaks
```

giving each peak its own position and width. The number of peaks is chosen by
BIC, so an extra peak has to earn its parameters, and fitted components that turn
out to be background rather than peaks are rejected (too wide, centre not located,
or overlapping a stronger peak). Curves with no resolvable peak say so.

On IPTS-38773 this resolves a reproducible **second peak at ~13.5 nm** alongside
the ~19 nm primary in every `leaf1_dark` curve — structure the earlier
single-peak treatment missed.

The fits themselves are shown in an appendix, **Peak Fit Evidence** — one compact
panel per sample in a grid (2 columns up to 4 curves, 3 beyond), each showing the
data, the background, the total fit and the fitted peak positions, annotated with
its repeat distance. The whole series lands on one page, so how the peak moves
from sample to sample reads at a glance. Every peak table cross-references that
figure, so the numbers stay traceable to the fits without the diagnostics
crowding out the results.

**Lamellar systems.** Only the *stack* models report a spacing:
`lamellar_stack_caille`, `lamellar_hg_stack_caille`, `lamellar_stack_paracrystal`
have a `d_spacing` parameter. Plain `lamellar` / `lamellar_hg` are single-lamella
form factors with none, and cannot answer a repeat-distance question however well
they fit. When the experiment targets a repeat distance, peak-bearing models are
**forced into the candidate set**, so a report can never say a model "fails to
capture the lamellar peak" without a lamellar model having been tried.

**Cross-check, always.** The fitting section quotes the measured peak positions
next to the model verdict, and if the model's spacing matches no observed peak
(nor a low-order reflection of one) the report says so and tells the reader not to
quote it. That case is real: on IPTS-38773 the winning lamellar fit reports
d = 112 A while the measured peaks are at 192 and 137 A.

## Short names for long filenames

EQSANS output names encode the whole reduction, which overflows table columns and
swamps plot legends:

```
merged_leaf1_370_70_4m_2p5A_30Hz_4m2.5a30hz_frame0_4m2.5a30hz_frame1_Iq.txt
```

When names are long, the report derives a short label for each by dropping the
leading and trailing parts **every** file shares — what remains is exactly the
part that distinguishes the samples:

```
merged_leaf1_370_70_4m_2p5A_30Hz_...   ->  leaf1_370_70
merged_leaf2_dark457_78_4m_2p5A_30Hz_... ->  leaf2_dark457_78
```

Short names are used in every table, figure legend, section title, caption and
fit label, and a **Short Name Legend** appendix table maps each one back to its
full file, so nothing is lost.

It is deliberately all-or-nothing per experiment — a report never mixes derived
names with opaque `S7` labels. Names of 15 characters or fewer are never touched;
if no shared prefix/suffix can be dropped, names up to 30 characters are left
alone (a legible long name beats an opaque short one) and only beyond that are
indexed labels used.

## Versioning

Every report records the `reportit` version that produced it — on the title
page, in the footer of every page, at the top of `sasfit_notebook.md`, and in the
CLI output. `reportit --version` prints it.

The version lives in **one** place, `src/reportit/__init__.py`; `pyproject.toml`
reads it from there, so the two cannot drift.

| Change | Bump |
|---|---|
| small change — bug fix, wording, tuning, a config/model swap | **+0.0.1** |
| major function addition — a new capability or CLI flag | **+0.1.0** |

When you change the code, bump `__version__` and add a line to
[CHANGELOG.md](CHANGELOG.md).

## Related projects

Reuses patterns from the EQSANS shared scripts: `../eqsanstools-cli`
(ONCat + LLM + plotting) and `../eqsanstools` (reduction + `eqplot`).
