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
OPENROUTER_MODEL=google/gemini-3.5-flash
```

The `.env` is gitignored — never commit it.

## CLI options

| Flag | Effect |
|------|--------|
| `-o, --out DIR` | output directory (default `./reportit_out/IPTS-<n>`) |
| `--strategy-only` | print the LLM-derived `AnalysisStrategy` JSON and stop |
| `--no-llm` | deterministic mode: heuristic grouping, no LLM reasoning |
| `--no-proposal` | ignore the proposal PDF(s) |
| `--refresh` | bust caches (re-query ONCat / re-run LLM) |
| `--max-llm-steps N` | cap on agentic strategy tool-calling steps (default 40) |
| `-v, --verbose` | verbose logging (shows each strategy probe) |

## Graceful degradation

A thin/image-only proposal, missing ONCat, or `--no-llm` never hard-fails: the
report falls back to a data-driven summary (heuristic grouping, ONCat/ filename
titles, templated observations) and records what was missing in a Caveats section.

## Related projects

Reuses patterns from the EQSANS shared scripts: `../eqsanstools-cli`
(ONCat + LLM + plotting) and `../eqsanstools` (reduction + `eqplot`).
