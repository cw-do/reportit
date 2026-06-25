"""Load reference knowledge (general SANS data-analysis / model-selection guidance)
for the LLM model-selector. NOT experiment-specific.

Drop reference files — Markdown, text, OR PDF (e.g. a SANS data-analysis article)
— into any of these locations (all are concatenated):
  - $REPORTIT_KNOWLEDGE_DIR
  - ~/.reportit/knowledge/
  - <repo>/knowledge/            (ships with a starter guide: sans_model_selection.md)

PDFs are text-extracted with pypdf. Large docs are truncated to keep prompts
sane; for very large corpora, prefer splitting into focused notes (future work:
embedding-based retrieval of the most relevant sections per query).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# generous — the models have large context windows; don't starve the selector
_MAX_CHARS = 200000
_PER_DOC_CHARS = 120000
_cache: str | None = None


def _dirs() -> list[Path]:
    dirs = []
    env = os.getenv("REPORTIT_KNOWLEDGE_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(Path.home() / ".reportit" / "knowledge")
    dirs.append(Path(__file__).resolve().parent.parent.parent.parent / "knowledge")
    return dirs


def knowledge_dirs() -> list[Path]:
    """Public: the directories searched for reference docs (for messaging)."""
    return _dirs()


def _read_doc(f: Path) -> str:
    if f.suffix.lower() in (".md", ".txt"):
        return f.read_text(errors="replace")
    if f.suffix.lower() == ".pdf":
        try:
            from ..proposal.extract import extract_text
            return extract_text(f)
        except Exception as e:  # noqa: BLE001
            logger.warning("knowledge PDF extract failed %s: %s", f, e)
            return ""
    return ""


def load_knowledge(refresh: bool = False) -> str:
    """Concatenate all reference docs found (md/txt/pdf), truncated to a sane size."""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    parts: list[str] = []
    seen: set[str] = set()
    for d in _dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix.lower() not in (".md", ".txt", ".pdf"):
                continue
            if f.name in seen:
                continue
            seen.add(f.name)
            text = _read_doc(f).strip()
            if not text:
                continue
            if len(text) > _PER_DOC_CHARS:
                text = text[:_PER_DOC_CHARS] + "\n...[doc truncated]"
            parts.append(f"# {f.name}\n{text}")

    text = "\n\n".join(parts).strip()
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n...[knowledge truncated]"
    _cache = text
    return text
