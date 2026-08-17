"""Command-line interface: reportit <ipts-or-path> [options]."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import __version__, pipeline


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="reportit")
@click.argument("target", required=False)
@click.option("-o", "--out", "out_dir", default=None,
              help="Output directory (default: ./reportit_out).")
@click.option("--no-llm", is_flag=True, help="Deterministic mode — no LLM reasoning.")
@click.option("--proposal", "proposal_path", type=click.Path(), default=None,
              metavar="PATH",
              help="Proposal document folder or a single PDF to use. If omitted, "
                   "proposals are auto-discovered from the experiment's shared "
                   "folder: '<shared>/proposal/*.pdf' (the default location), plus "
                   "any other *.pdf found under '<shared>' (e.g. relevant prior "
                   "work). Use --no-proposal to ignore proposals entirely.")
@click.option("--no-proposal", is_flag=True, help="Ignore the proposal PDF(s).")
@click.option("--data", "data_dirs", type=click.Path(), multiple=True, metavar="DIR",
              help="Reduced-data folder to analyze. If omitted, folders are "
                   "auto-discovered as every directory under TARGET that holds "
                   "reduced 1D data ('*_Iq.dat' plus merged/stitched profiles). "
                   "Repeat the flag to analyze several reductions as variants. "
                   "The folder may lie outside TARGET, and an explicit choice "
                   "overrides the LLM's variant selection.")
@click.option("--userguide", "--guide", "user_guide", default=None, metavar="TEXT",
              help="A few sentences steering the analysis, in plain English "
                   '(e.g. "use merged files only", "skip the porsil standard", '
                   '"group by salinity, not temperature"). It is interpreted once '
                   "and routed to the steps it actually affects: selection rules "
                   "become a concrete dataset list handed to every later stage, "
                   "and advisory parts are passed to the strategy / fitting / "
                   "narrative steps. What it did is recorded in the report.")
@click.option("--knowledge", "knowledge_dirs", type=click.Path(), multiple=True,
              metavar="DIR",
              help="Extra directory of reference-knowledge notes (.md/.txt/.pdf) to "
                   "use for this run, on top of $REPORTIT_KNOWLEDGE_DIR, "
                   "~/.reportit/knowledge/ and the guide shipped with reportit. "
                   "Repeatable.")
@click.option("--learn", "learn_text", default=None, metavar="TEXT",
              help="Teach reportit a general lesson and exit. The text is appended "
                   "to ~/.reportit/knowledge/lessons.md and is read by EVERY later "
                   "run, for every experiment. Pair with --learn-stage / "
                   "--learn-title to say where it applies.")
@click.option("--learn-stage", default="all", metavar="STAGES", show_default=True,
              help="Which stage(s) a --learn lesson applies to: strategy, fitting, "
                   "critic, narrative, or all (comma-separated).")
@click.option("--learn-title", default="", metavar="TEXT",
              help="Short title for a --learn lesson.")
@click.option("--show-knowledge", is_flag=True,
              help="List the reference-knowledge notes reportit will use, per "
                   "stage, and exit.")
@click.option("--strategy-only", is_flag=True,
              help="Print the LLM-derived AnalysisStrategy and stop.")
@click.option("--refresh", is_flag=True, help="Bust caches (re-query ONCat/LLM).")
@click.option("--sasfit/--no-sasfit", default=True,
              help="Agentic sasmodels model-based fitting (model selection + bumps "
                   "fit + critic loop, then fit every member) per group. On by "
                   "default; use --no-sasfit for a quick run without it.")
@click.option("--summary-only", is_flag=True,
              help="Generate only report_summary.pdf (skip the comprehensive "
                   "report). Pair with --no-sasfit for the fastest overview run.")
@click.option("--max-llm-steps", type=int, default=None,
              help="Max agentic strategy tool-calling steps.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def main(target, out_dir, no_llm, proposal_path, no_proposal, data_dirs, user_guide,
         knowledge_dirs, learn_text, learn_stage, learn_title, show_knowledge,
         strategy_only, refresh, sasfit, summary_only, max_llm_steps, verbose):
    """Generate an EQSANS post-experiment report for an IPTS number or shared path."""
    logging.basicConfig(
        level=logging.INFO if verbose or strategy_only else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Always show our own pipeline INFO lines for progress.
    logging.getLogger("reportit").setLevel(logging.INFO)

    from .analysis import knowledge
    if knowledge_dirs:
        knowledge.add_dirs(knowledge_dirs)

    # Teaching mode: record a lesson for every future run, then stop.
    if learn_text:
        path = knowledge.learn(learn_text, title=learn_title,
                               applies_to=learn_stage)
        click.echo(f"Learned. Appended to {path}")
        click.echo("It will be used by every later run (add --refresh on the first "
                   "rerun of an experiment you have already analysed).")
        return

    if show_knowledge:
        click.echo("Knowledge directories searched (first match of a filename wins):")
        for d in knowledge.knowledge_dirs():
            click.echo(f"  {'[x]' if d.is_dir() else '[ ]'} {d}")
        notes = knowledge.load_notes(refresh=True)
        click.echo(f"\n{len(notes)} note(s) found:")
        for n in notes:
            click.echo(f"  {n.name}  ({len(n.text)} chars)"
                       f"  applies_to={','.join(n.applies_to)}"
                       f"{'  [always]' if n.always else ''}")
            click.echo(f"      title: {n.title}")
        click.echo("\nPer stage:")
        for st in knowledge.STAGES:
            used = knowledge.used_names(st)
            click.echo(f"  {st:10s} {len(used)} note(s): {', '.join(used) or '(none)'}")
        return

    if not target:
        raise click.UsageError("Missing argument 'TARGET' (an IPTS number or a path). "
                               "Use --learn/--show-knowledge without a target.")

    if out_dir is None:
        out_dir = Path.cwd() / "reportit_out"

    try:
        result = pipeline.run_report(
            target, out_dir,
            no_llm=no_llm, no_proposal=no_proposal, proposal_path=proposal_path,
            data_dirs=list(data_dirs), user_guide=user_guide,
            strategy_only=strategy_only, refresh=refresh,
            sasfit=sasfit, summary_only=summary_only, max_llm_steps=max_llm_steps,
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"error: {e}", err=True)
        if verbose:
            raise
        sys.exit(1)

    if strategy_only:
        return

    click.echo("")
    click.echo(f"reportit v{__version__}")
    click.echo(f"Output dir: {result.out_dir}")
    for p in result.tex_files:
        click.echo(f"  tex: {p}")
    for p in result.pdfs:
        click.echo(f"  pdf: {p}")
    if not result.pdfs:
        click.echo("  (no PDF — pdflatex unavailable or failed; see .tex/.log)")


if __name__ == "__main__":
    main()
