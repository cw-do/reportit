"""Deterministic, bounded folder walk producing a compact FolderInventory digest.

The digest (not the 1400 raw paths) is what the LLM strategist first receives.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from pathlib import Path

from ..models import FileEntry, FolderInventory
from . import scan

logger = logging.getLogger(__name__)

# Directories we never descend into (huge / irrelevant).
_SKIP_DIRS = {".reportit_cache", "__pycache__", ".git", ".ipynb_checkpoints"}

_IQ1D_RE = re.compile(r"_Iq\.dat$", re.IGNORECASE)
_IQ2D_RE = re.compile(r"_Iqxqy\.(dat|h5)$", re.IGNORECASE)
_NOTE_RE = re.compile(r"(note|readme).*", re.IGNORECASE)


def _classify(path: Path) -> str:
    name = path.name
    ext = path.suffix.lower()
    if ext == ".pdf" and ("proposal" in str(path).lower() or "proposal" in name.lower()
                          or "review" in name.lower() or "sor" in name.lower()):
        return "proposal"
    if ext == ".pdf":
        return "proposal"  # any PDF in shared is a candidate proposal/literature
    if ext == ".py":
        return "script"
    if scan.is_combined_name(name):
        return "combined"
    if _IQ1D_RE.search(name):
        return "iq1d"
    if _IQ2D_RE.search(name):
        return "iqxqy2d"
    if ext == ".json":
        return "reduction_json"
    if _NOTE_RE.match(name) and ext in (".md", ".txt", ""):
        return "note"
    if name.lower().endswith("_trans.txt"):
        return "trans"
    if ext in (".png", ".jpg", ".jpeg"):
        return "image"
    if ext in (".nxs", ".h5", ".hdf"):
        return "nexus"
    return "other"


def _build_tree_text(root: Path, max_depth: int = 3, max_entries: int = 40) -> str:
    """A compact directory tree (dirs + file-count summary per dir)."""
    lines: list[str] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        rel = Path(dirpath).resolve().relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        label = root.name if str(rel) == "." else rel.parts[-1]
        ext_counter = Counter(Path(f).suffix.lower() or "<none>" for f in filenames)
        summary = ", ".join(f"{n}{ext}" for ext, n in ext_counter.most_common(6))
        lines.append(f"{indent}{label}/  ({len(filenames)} files: {summary})")
        if len(lines) > max_entries:
            lines.append("  ... (tree truncated)")
            break
    return "\n".join(lines)


def build(target: str | int | Path, max_files: int = 20000) -> FolderInventory:
    """Build a FolderInventory for an IPTS number, IPTS-NNNNN, or a path."""
    shared_dir, ipts = resolve_target(target)

    entries: list[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(shared_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            entries.append(FileEntry(path=p, size=size, ext=p.suffix.lower(),
                                     kind=_classify(p)))
            if len(entries) >= max_files:
                break
        if len(entries) >= max_files:
            break

    ext_counts = Counter(e.ext or "<none>" for e in entries)
    kind_counts = Counter(e.kind for e in entries)

    proposal_pdfs = [e.path for e in entries if e.kind == "proposal"]
    scripts = [e.path for e in entries if e.kind == "script"]
    note_files = [e.path for e in entries if e.kind == "note"]

    # Candidate output dirs = directories that directly contain reduced 1D data.
    output_dirs = sorted({e.path.parent for e in entries
                          if e.kind in ("iq1d", "combined")})

    # Representative output names (from 1D files), capped.
    iq_names = sorted({_IQ1D_RE.sub("", e.path.name) for e in entries if e.kind == "iq1d"})
    naming_examples = iq_names[:40]

    # Combined/stitched 1D profiles (naming varies — merged/stitched/etc.).
    combined_examples = sorted({e.path.name for e in entries if e.kind == "combined"})[:20]

    tree_text = _build_tree_text(shared_dir)

    return FolderInventory(
        ipts=ipts,
        shared_dir=shared_dir,
        tree_text=tree_text,
        ext_counts=dict(ext_counts),
        kind_counts=dict(kind_counts),
        output_dirs=output_dirs,
        proposal_pdfs=proposal_pdfs,
        scripts=scripts,
        note_files=note_files,
        naming_examples=naming_examples,
        combined_examples=combined_examples,
        total_files=len(entries),
    )


def resolve_target(target: str | int | Path) -> tuple[Path, int]:
    """Resolve an IPTS number / 'IPTS-NNNNN' / a path into (shared_dir, ipts)."""
    s = str(target).strip()

    # A path?
    p = Path(s)
    if p.exists() and p.is_dir():
        shared = p if p.name == "shared" else (p / "shared" if (p / "shared").is_dir() else p)
        m = re.search(r"IPTS-(\d+)", str(p.resolve()))
        ipts = int(m.group(1)) if m else 0
        return shared.resolve(), ipts

    # An IPTS number or IPTS-NNNNN
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"Could not resolve target: {target!r}")
    ipts = int(m.group(1))
    for base in ("/SNS/EQSANS", "/gpfs/neutronsfs/instruments/EQSANS"):
        cand = Path(base) / f"IPTS-{ipts}" / "shared"
        if cand.is_dir():
            return cand.resolve(), ipts
    raise FileNotFoundError(f"No shared dir found for IPTS-{ipts}")
