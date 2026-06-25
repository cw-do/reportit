# reportit — Task Tracker

Status legend: [x] done · [~] partial · [ ] todo

## Foundation
- [x] `pyproject.toml`, package skeleton, venv (python3.11), editable install
- [x] `config.py` — load OpenRouter/LLM settings from `.env`
- [x] `models.py` — dataclasses passed between stages
- [x] `cache/store.py` — md5-keyed JSON disk cache

## Discovery
- [x] `discovery/inventory.py` — bounded folder walk → `FolderInventory` digest
- [x] `discovery/naming.py` — `[BASE]_[TEMP?]_[CONFIG]` parser + standard detection
- [x] `discovery/reduction_json.py` — parse per-sample reduction config
- [x] `discovery/scan.py` — build `Dataset` records (1D/2D/merged/json siblings)

## Integrations
- [x] `integrations/oncat.py` — `fetch_catalog` (pyoncat) + disk cache

## LLM core (the heart)
- [x] `llm/client.py` — `chat`, `chat_json`, `chat_with_tools` (agentic loop,
      fallback model, escalating wrap-up nudge, robust forced finalize, caching)
- [x] `llm/tools.py` — probe tool specs + `finalize_strategy` schema
- [x] `strategy/probes.py` — read-only probes (list_dir, read_text, head_file,
      parse_reduction_json, oncat_titles, sample_curve, list_datasets), sandboxed
- [x] `strategy/engine.py` — agentic strategy loop + deterministic fallback

## Proposal
- [x] `proposal/extract.py` — pypdf text extraction (+ pdfplumber fallback)
- [x] `proposal/summarize.py` — LLM → structured `ProposalInfo` + hypotheses

## Analysis / plotting
- [x] `analysis/loaders.py` — native loaders incl. tolerant `Iqxqy` parser
- [x] `analysis/metrics.py` — q-range, log-log slopes, flags
- [x] `analysis/fit.py` — Guinier + power-law/Porod (scipy)
- [x] `plotting/figures.py` — log-log 1D overlay (+fit, +variant compare), 2D map

## Execute / narrative / report
- [x] `execute/runner.py` — strategy → figures, metrics tables, fits
- [x] `narrative/synthesize.py` — per-group observations + global discussion +
      hypothesis checks (LLM, with deterministic fallback)
- [x] `report/latex_utils.py` — LaTeX escaping + Unicode sanitization
- [x] `report/templates/report.tex.j2` — generic `article` template
- [x] `report/assemble.py` — jinja2 render (comprehensive + summary)
- [x] `report/compile.py` — pdflatex ×2, degrade to `.tex` if absent
- [x] `pipeline.py`, `cli.py`, `__main__.py`

## Verified on IPTS-38533
- [x] inventory/scan/metrics on real data
- [x] deterministic `--no-llm` → 2 PDFs compile
- [x] agentic `--strategy-only` → LLM correctly identified the science
      (d-P2VPNO polyzwitterions), chose `output` variant w/ rationale, built
      temperature + concentration series, planned Guinier fits
- [x] full run → comprehensive (10pp) + summary PDFs with figures, fits,
      hypothesis assessment, discussion
- [x] 2D `I(Qx,Qy)` plotting path

## Tables + fitting improvements (user feedback round 1)
- [x] Split Sample Summary (sample/description/configs/conditions) from run tables
- [x] Appendix: landscape Reduction Run Table (scatt/trans per config + bkg/empty)
- [x] Appendix: ONCat Run Catalog (run/title/distance/wavelength/counts/duration)
- [x] Ornstein-Zernike correlation-length fit; LLM-chosen fit model; low-Q outlier trim
- [x] Fix pdflatex output utf-8 decode crash

## Agentic sasmodels fitting (user feedback round 2)
- [x] `--sasfit` mode: model-select (glm-5.2) -> bumps fit -> critic (gemini vision
      + glm reasoning) -> iterate -> report success/failure
- [x] `analysis/sascatalog.py` (catalog from sasmodels), `sasfit.py` (bumps fit),
      `sas_agent.py` (the loop)
- [x] Hybrid models: reasoning=z-ai/glm-5.2, vision=google/gemini-3.5-flash
- [x] Partial-Q-range fitting (exclude low-Q aggregation; critic accepts
      limited-range fits); excluded points drawn faintly
- [x] chat_json retries same model w/ correction on malformed JSON (+ trailing-comma
      strip); per-call model override has fallback chain
- [x] Report "Model-Based Fitting (sasmodels)" section + summary table

## Backlog / possible improvements
- [ ] Compute group metrics on the merged extended-Q curve (table q-max currently
      reflects the single-config member, not the merged curve used in the figure)
- [ ] Cover remaining ungrouped samples (banjo, pb30.*, D2O, ...) or note as out-of-scope
- [ ] `--variant` override flag to force a specific output dir
- [ ] Per-dataset (not just per-group representative) sasmodels fitting + trend tables
      (e.g. Rg vs temperature) once a model is accepted
- [ ] Parameter uncertainties: bumps `lm` returns stderr=0; run a DREAM/MCMC pass
      for error bars on accepted fits
- [ ] Unit tests + ruff in CI; parallelize per-group LLM calls for speed
