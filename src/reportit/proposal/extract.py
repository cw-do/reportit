"""Extract text from proposal PDF(s) using pypdf (pure-python).

Falls back to pdfplumber if installed and pypdf yields little text.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(pdf_path: str | Path) -> str:
    pdf_path = Path(pdf_path)
    text = _extract_pypdf(pdf_path)
    if len(text.strip()) < 200:
        plumber = _extract_pdfplumber(pdf_path)
        if len(plumber.strip()) > len(text.strip()):
            text = plumber
    return text


def extract_many(paths: list[Path]) -> str:
    parts = []
    for p in paths:
        try:
            t = extract_text(p)
        except Exception as e:  # noqa: BLE001
            logger.warning("PDF extract failed for %s: %s", p, e)
            t = ""
        if t.strip():
            parts.append(f"===== {p.name} =====\n{t}")
    return "\n\n".join(parts)


def _extract_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:  # noqa: BLE001
        logger.warning("pypdf failed on %s: %s", pdf_path, e)
        return ""


def _extract_pdfplumber(pdf_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            return "\n".join((pg.extract_text() or "") for pg in pdf.pages)
    except Exception as e:  # noqa: BLE001
        logger.warning("pdfplumber failed on %s: %s", pdf_path, e)
        return ""
