"""Command-line interface: reportit <ipts-or-path> [options]."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import pipeline


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("target")
@click.option("-o", "--out", "out_dir", default=None,
              help="Output directory (default: ./reportit_out/IPTS-<n>).")
@click.option("--no-llm", is_flag=True, help="Deterministic mode — no LLM reasoning.")
@click.option("--no-proposal", is_flag=True, help="Ignore the proposal PDF(s).")
@click.option("--strategy-only", is_flag=True,
              help="Print the LLM-derived AnalysisStrategy and stop.")
@click.option("--refresh", is_flag=True, help="Bust caches (re-query ONCat/LLM).")
@click.option("--sasfit", is_flag=True,
              help="Run agentic sasmodels model-based fitting (model selection + "
                   "bumps fit + critic loop) per group.")
@click.option("--max-llm-steps", type=int, default=None,
              help="Max agentic strategy tool-calling steps.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def main(target, out_dir, no_llm, no_proposal, strategy_only, refresh, sasfit, max_llm_steps, verbose):
    """Generate an EQSANS post-experiment report for an IPTS number or shared path."""
    logging.basicConfig(
        level=logging.INFO if verbose or strategy_only else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Always show our own pipeline INFO lines for progress.
    logging.getLogger("reportit").setLevel(logging.INFO)

    if out_dir is None:
        try:
            _, ipts = pipeline.inventory.resolve_target(target)
        except Exception:
            ipts = 0
        out_dir = Path.cwd() / "reportit_out" / (f"IPTS-{ipts}" if ipts else "report")

    try:
        result = pipeline.run_report(
            target, out_dir,
            no_llm=no_llm, no_proposal=no_proposal,
            strategy_only=strategy_only, refresh=refresh,
            sasfit=sasfit, max_llm_steps=max_llm_steps,
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"error: {e}", err=True)
        if verbose:
            raise
        sys.exit(1)

    if strategy_only:
        return

    click.echo("")
    click.echo(f"Output dir: {result.out_dir}")
    for p in result.tex_files:
        click.echo(f"  tex: {p}")
    for p in result.pdfs:
        click.echo(f"  pdf: {p}")
    if not result.pdfs:
        click.echo("  (no PDF — pdflatex unavailable or failed; see .tex/.log)")


if __name__ == "__main__":
    main()
