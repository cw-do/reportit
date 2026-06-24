# reportit

Automated **post-experiment report generator** for EQSANS (Extended Q-range
Small-Angle Neutron Scattering) experiments at the SNS, Oak Ridge National
Laboratory.

## Goal

Given an experiment's IPTS folder (`/SNS/EQSANS/IPTS-{num}/shared/`), `reportit`
produces a LaTeX/PDF experimental report that serves both as a **summary** of the
science and as a **record** of what data was collected and how it was reduced.

The intended pipeline:

1. **Read the proposal** — ingest one or more PDFs under `shared/proposal/`
   (beamtime proposal plus relevant prior work / literature) and use an LLM to
   understand what data the experiment expects to produce and what trends were
   hypothesized.
2. **Inventory the data** — walk `shared/` and its subfolders to find reduction
   scripts and reduced output (`*_Iq.dat` 1D, `*_Iqxqy.dat` 2D), noting which
   folder holds which data.
3. **Identify runs** — use ONCat (see the `eqsanstools` / `eqsanstools-cli`
   projects for working code) to map run numbers to data types based on run
   titles.
4. **Reason** — use an LLM to best-guess what each dataset represents by
   combining proposal context with run titles (which often contain
   abbreviations), and form a strategy for which data to group/compare and
   whether 1D or 2D comparison is meaningful.
5. **Plot** — generate figures (1D `Iq.dat` as log-log, with accurate, relevant
   Q-range labels; 2D where appropriate), writing fit functions via LLM as
   needed.
6. **Write the report** — assemble observations (including whether hypothesized
   trends appear) into a LaTeX document and compile it to PDF.

When sample information is insufficient, the tool degrades gracefully: at minimum
it summarizes what ONCat and the available data show, plotting by run title.

## Configuration

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` (OpenRouter is used
for the LLM reasoning steps). The `.env` file is gitignored — never commit it.

## Related projects

This repo lives alongside the EQSANS shared scripts. Reusable ONCat / reduction
logic to draw from:

- `../eqsanstools` — core reduction tooling (`eqsans_drtsans_script.py`,
  catalog/ONCat utilities) built on the `drtsans` package.
- `../eqsanstools-cli` — interactive Textual TUI/CLI for EQSANS reduction
  (catalog loading, run matching, reduction, plotting, stitching).

## Status

Early scaffolding. The detailed implementation plan is being prepared
separately.
