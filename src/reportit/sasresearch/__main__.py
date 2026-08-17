"""CLI: ``python -m reportit.sasresearch <datafile> [options]``.

Standalone entry point for developing the SAS fitting autoresearch loop without
running the full report pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..cache import Cache
from ..config import AppSettings
from ..llm import LLMClient
from . import loop


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="reportit.sasresearch",
        description="Autoresearch SANS fitting explorer for one I(Q) curve.")
    p.add_argument("datafile", help="1D I(Q) data file (_Iq.dat or merged *_Iq.txt)")
    p.add_argument("--out", default=None,
                   help="output directory (default: ./sasresearch_out/<datafile stem>)")
    p.add_argument("--models", default=None,
                   help="comma-separated model list to try (skips LLM selection)")
    p.add_argument("--context", default="",
                   help="free-text experiment context to guide model selection")
    p.add_argument("--n-starts", type=int, default=8,
                   help="sampled starting points per model (default 8)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip all LLM calls: use the default model shortlist and no vision check")
    p.add_argument("--no-refine", action="store_true",
                   help="skip the window-refinement step")
    p.add_argument("--refresh", action="store_true", help="ignore cached LLM responses")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    datafile = Path(args.datafile)
    if not datafile.is_file():
        print(f"error: no such file: {datafile}", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else Path.cwd() / "sasresearch_out" / datafile.stem

    llm = None
    if not args.no_llm:
        settings = AppSettings.load()
        if settings.llm.is_configured:
            cache = Cache(out_dir / ".sasresearch_cache", enabled=True, bust=args.refresh)
            llm = LLMClient(settings.llm, cache=cache)
        else:
            print("warning: no OPENROUTER_API_KEY; running without LLM "
                  "(default model shortlist, no vision check).", file=sys.stderr)

    models = [m.strip() for m in args.models.split(",")] if args.models else None

    result = loop.explore(
        datafile, out_dir, llm=llm, context=args.context, models=models,
        n_starts=args.n_starts, use_llm=llm is not None, refine=not args.no_refine)

    print(f"\nBest model: {result['winner']}")
    print("Ranked:")
    for e in result["ranked"]:
        mark = "ok " if e.get("ok") else "FAIL"
        print(f"  [{mark}] {e['model']:<22} score={e.get('score', 0):.3f}"
              + (f"  red-chi2={e.get('reduced_chisq'):.2f}" if e.get("reduced_chisq") else ""))
    print(f"\nArtefacts in: {out_dir}")
    print(f"  notebook.md, results.json, best_fit.png, candidates/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
