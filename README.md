# reportit

**LLM-driven post-experiment report generator for EQSANS** (Extended Q-range
Small-Angle Neutron Scattering) at the SNS, Oak Ridge National Laboratory.

Point it at an experiment's IPTS folder and it reads the proposal, figures out
what the experiment was about, inventories the reduced data, and writes a
generic-format LaTeX report (compiled to PDF) summarizing the science, showing
plots, and assessing whether the proposal's hypotheses are borne out by the data.

```bash
reportit 38533                 # -> ./reportit_out/IPTS-38533/report_*.pdf
reportit /SNS/EQSANS/IPTS-38533/shared -o /tmp/rep
```

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
inventory ─▶ proposal (pypdf + LLM) ─▶ STRATEGY (agentic LLM + probes)
          ─▶ execute (load, metrics, fits, plots) ─▶ narrative (LLM)
          ─▶ assemble LaTeX ─▶ pdflatex ×2 ─▶ report_{comprehensive,summary}.pdf
```

Key modules (`src/reportit/`): `discovery/` (folder inventory, name parsing,
reduction-JSON), `integrations/oncat.py` (run catalog via pyoncat),
`proposal/` (PDF text + LLM summary), `llm/` (OpenRouter client with
caching, JSON, and the tool-calling loop; probe tool specs), `strategy/`
(the agentic engine + read-only probes), `analysis/` (native numpy loaders,
metrics, scipy fits — **no drtsans dependency**), `plotting/figures.py`,
`execute/runner.py`, `narrative/synthesize.py`, `report/` (jinja2 templates +
pdflatex). Every ONCat / LLM / probe result is cached under
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
OPENROUTER_MODEL=google/gemini-3.5-flash            # strategy + narrative
OPENROUTER_REASONING_MODEL=z-ai/glm-5.2             # model selection / fit critique
OPENROUTER_VISION_MODEL=google/gemini-3.5-flash     # visually inspect fit plots
```

The `.env` is gitignored — never commit it.

## Model-based fitting (on by default; `--no-sasfit` to skip)

Every run does an agentic, sasmodels-based fitting stage (it's the core of the
analysis). Use `--no-sasfit` for a quick run without it. Per sample group it runs
a multi-agent loop:

1. a **reasoning agent** (glm-5.2) selects candidate SasView/sasmodels models and
   a parameter plan (initial guesses, which params to fit vs fix, bounds, and an
   optional Q-window) from the curve shape + proposal context;
2. each candidate is fit with **bumps** (`sasmodels` + `bumps`);
3. a **critic** judges it — a multimodal model (gemini) visually inspects the
   fit-vs-data plot, and the reasoning model evaluates χ², residuals, and
   parameter sanity — then accepts or rejects and the loop iterates;
4. once a model is chosen, **every member of the group is fit with it**, and the
   report tabulates and plots the trend of the key parameter across the series
   (e.g. correlation length or Rg vs temperature);
5. the report's "Model-Based Fitting" section shows the chosen model, fitted
   parameters, the fit figure, the per-member trend, and the critic's verdict —
   **including honest failures**.

Partial-Q-range fits are first-class: a low-Q upturn (aggregation / large-scale
structure outside the length scale of interest) can be excluded so the model is
fit only over the regime it applies to; excluded points are shown faintly.

## CLI options

| Flag | Effect |
|------|--------|
| `-o, --out DIR` | output directory (default `./reportit_out/IPTS-<n>`) |
| `--strategy-only` | print the LLM-derived `AnalysisStrategy` JSON and stop |
| `--no-llm` | deterministic mode: heuristic grouping, no LLM reasoning |
| `--no-proposal` | ignore the proposal PDF(s) |
| `--refresh` | bust caches (re-query ONCat / re-run LLM) |
| `--no-sasfit` | skip the agentic sasmodels model-based fitting (on by default) |
| `--max-llm-steps N` | cap on agentic strategy tool-calling steps (default 40) |
| `-v, --verbose` | verbose logging (shows each strategy probe) |

## Graceful degradation

A thin/image-only proposal, missing ONCat, or `--no-llm` never hard-fails: the
report falls back to a data-driven summary (heuristic grouping, ONCat/ filename
titles, templated observations) and records what was missing in a Caveats section.

## Related projects

Reuses patterns from the EQSANS shared scripts: `../eqsanstools-cli`
(ONCat + LLM + plotting) and `../eqsanstools` (reduction + `eqplot`).
