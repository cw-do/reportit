"""Compile a .tex to PDF with pdflatex (run twice). Degrade to .tex if absent."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def has_pdflatex() -> bool:
    return shutil.which("pdflatex") is not None


def compile_pdf(tex_path: Path) -> Path | None:
    """Compile tex_path → PDF in the same dir. Returns PDF path, or None on failure."""
    tex_path = Path(tex_path)
    out_dir = tex_path.parent
    if not has_pdflatex():
        logger.warning("pdflatex not found — wrote %s only (compile elsewhere).", tex_path.name)
        return None

    cmd = [
        "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
        f"-output-directory={out_dir}", tex_path.name,
    ]
    ok = True
    for _ in range(2):  # twice for refs/labels/longtable
        try:
            proc = subprocess.run(cmd, cwd=str(out_dir), capture_output=True,
                                  text=True, errors="replace", timeout=180)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("pdflatex invocation failed: %s", e)
            ok = False
            break
        if proc.returncode != 0:
            ok = False  # keep going to first pass log; report below

    pdf = tex_path.with_suffix(".pdf")
    if pdf.is_file():
        if not ok:
            logger.warning("pdflatex reported errors but a PDF was produced (%s).", pdf.name)
        return pdf
    logger.warning("pdflatex failed; see %s", tex_path.with_suffix(".log"))
    return None
