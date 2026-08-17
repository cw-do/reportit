"""Reference knowledge — the part of reportit you teach.

Drop notes into a knowledge directory and every later run reads them. Nothing
here is experiment-specific: these are the rules and habits that should apply to
the NEXT experiment too.

Where notes are read from (all are merged; later dirs do not override earlier
ones, they add to them):

  - any directory passed with ``--knowledge DIR``
  - ``$REPORTIT_KNOWLEDGE_DIR``
  - ``~/.reportit/knowledge/``          <- personal notes; `--learn` writes here
  - ``<repo>/knowledge/``               <- the curated guide that ships with reportit

Markdown, plain text, or PDF (a SANS paper can be dropped in as-is and is
text-extracted).

A note may declare, in an optional front-matter block, WHICH pipeline stage it is
for and WHAT it is about::

    ---
    title: Lamellar systems need a peak model
    applies_to: fitting, critic
    keywords: lamellar, d-spacing, peak, stacking
    priority: always
    ---
    Body of the lesson...

All fields are optional — a plain Markdown file with no front matter is treated
as general guidance for every stage, which is why existing notes keep working.

**Stage routing.** ``applies_to`` decides which prompt a note reaches:
``strategy`` (how to read the folder, group samples, decide what to analyse),
``fitting`` (which models to try), ``critic`` (whether a fit is acceptable),
``narrative`` (how to write the report), or ``all``.

**Relevance.** While the corpus is small everything is sent, exactly as before.
Once it outgrows the budget for a stage, notes are ranked against the actual
experiment — its proposal text, curve shapes and candidate models — and the most
relevant are sent. Notes marked ``priority: always`` are never dropped. This is
deliberately keyword-based and deterministic: no embedding service, no extra
dependency, and the same experiment always gets the same knowledge.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STAGES = ("strategy", "fitting", "critic", "narrative")

# Per-stage budget in characters. The models have large context windows, so these
# are generous; they exist to stop a growing library from crowding out the actual
# data, not to save tokens.
_STAGE_BUDGET = {
    "strategy": 60000,
    "fitting": 120000,
    "critic": 60000,
    "narrative": 30000,
}
_PER_DOC_CHARS = 120000

_FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_WORD_RE = re.compile(r"[a-z0-9_]{3,}")

_extra_dirs: list[Path] = []
_cache: dict = {}


@dataclass
class Note:
    name: str
    path: Path
    text: str
    title: str = ""
    applies_to: tuple[str, ...] = ("all",)
    keywords: tuple[str, ...] = ()
    always: bool = False
    # filled per-run, for reporting which knowledge was actually used
    score: float = field(default=0.0, compare=False)

    def targets(self, stage: str) -> bool:
        return "all" in self.applies_to or stage in self.applies_to


# --------------------------------------------------------------------------- #
# where notes come from
# --------------------------------------------------------------------------- #
def add_dirs(dirs) -> None:
    """Register extra knowledge directories (from --knowledge) for this run."""
    global _cache
    for d in dirs or []:
        p = Path(d).expanduser()
        if p not in _extra_dirs:
            _extra_dirs.append(p)
    _cache = {}


def user_dir() -> Path:
    """The personal notes directory that `--learn` writes to."""
    return Path.home() / ".reportit" / "knowledge"


def _dirs() -> list[Path]:
    dirs = list(_extra_dirs)
    env = os.getenv("REPORTIT_KNOWLEDGE_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(user_dir())
    dirs.append(Path(__file__).resolve().parent.parent.parent.parent / "knowledge")
    return dirs


def knowledge_dirs() -> list[Path]:
    """Public: the directories searched for reference docs (for messaging)."""
    return _dirs()


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
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


def _split_list(v: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in re.split(r"[,;]", v) if x.strip())


def _parse_note(f: Path, raw: str) -> Note:
    title, applies, keys, always = "", ("all",), (), False
    m = _FRONT_RE.match(raw)
    body = raw
    if m:
        body = raw[m.end():]
        for line in m.group(1).splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip().lower(), v.strip()
            if k == "title":
                title = v
            elif k in ("applies_to", "stage", "stages"):
                vals = _split_list(v)
                applies = vals or ("all",)
            elif k == "keywords":
                keys = _split_list(v)
            elif k == "priority":
                always = v.strip().lower() in ("always", "high", "critical")
    if not title:
        h = re.search(r"^#\s+(.+)$", body, re.M)
        title = h.group(1).strip() if h else f.stem.replace("_", " ")
    if not keys:
        # derive keywords from the title and any headings — good enough to rank
        heads = " ".join(re.findall(r"^#{1,3}\s+(.+)$", body, re.M)[:40])
        keys = tuple(sorted(set(_WORD_RE.findall((title + " " + heads).lower()))))[:60]
    if len(body) > _PER_DOC_CHARS:
        body = body[:_PER_DOC_CHARS] + "\n...[doc truncated]"
    return Note(name=f.name, path=f, text=body.strip(), title=title,
                applies_to=applies, keywords=keys, always=always)


def load_notes(refresh: bool = False) -> list[Note]:
    """Every note found, parsed. Cached per process."""
    if not refresh and "notes" in _cache:
        return _cache["notes"]
    notes: list[Note] = []
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
            raw = _read_doc(f).strip()
            if not raw:
                continue
            notes.append(_parse_note(f, raw))
    _cache["notes"] = notes
    return notes


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _relevance(note: Note, context: str) -> float:
    if not context:
        return 0.0
    ctx = context.lower()
    hits = sum(1 for k in note.keywords if k and k in ctx)
    return hits / max(1, len(note.keywords)) * 10.0 + min(hits, 10)


def select(stage: str = "fitting", context: str = "", *,
           budget: int | None = None, refresh: bool = False) -> list[Note]:
    """The notes to send to ``stage`` for this experiment.

    Everything that targets the stage is returned while it fits the budget — so
    a small library behaves exactly as it always has. Only when the library grows
    past the budget does relevance ranking decide what to include, and notes
    marked ``priority: always`` are kept regardless.
    """
    budget = budget or _STAGE_BUDGET.get(stage, 60000)
    candidates = [n for n in load_notes(refresh) if n.targets(stage)]
    total = sum(len(n.text) for n in candidates)
    if total <= budget:
        return candidates

    for n in candidates:
        n.score = _relevance(n, context)
    ranked = sorted(candidates, key=lambda n: (n.always, n.score, -len(n.text)),
                    reverse=True)
    out, used = [], 0
    for n in ranked:
        if used + len(n.text) > budget and not n.always:
            continue
        out.append(n)
        used += len(n.text)
    logger.info("knowledge[%s]: %d of %d notes selected (%d/%d chars)",
                stage, len(out), len(candidates), used, budget)
    return out


def load_knowledge(refresh: bool = False, stage: str = "fitting",
                   context: str = "") -> str:
    """Knowledge text for ``stage``, ready to paste into a prompt."""
    notes = select(stage, context, refresh=refresh)
    if not notes:
        return ""
    return "\n\n".join(f"# {n.name}\n{n.text}" for n in notes).strip()


def digest(stage: str = "fitting", context: str = "") -> str:
    """Fingerprint of the knowledge a stage will actually receive.

    Mixed into the LLM cache keys so that TEACHING reportit something changes its
    answers: without this, a rerun in an existing output directory would replay
    the previous run's reasoning and the new lesson would appear to do nothing.
    """
    import hashlib
    notes = select(stage, context)
    h = hashlib.sha1()
    for n in sorted(notes, key=lambda x: x.name):
        h.update(n.name.encode())
        h.update(str(len(n.text)).encode())
        h.update(n.text[:4000].encode(errors="replace"))
    return h.hexdigest()[:10] if notes else "none"


def used_names(stage: str = "fitting", context: str = "") -> list[str]:
    """Names of the notes that would be used — recorded in the report."""
    return [n.name for n in select(stage, context)]


# --------------------------------------------------------------------------- #
# teaching
# --------------------------------------------------------------------------- #
def learn(text: str, *, title: str = "", applies_to: str = "all",
          keywords: str = "", target_dir: Path | None = None) -> Path:
    """Append a lesson to the personal knowledge notes and return the file path.

    Lessons are grouped into one file PER STAGE SET, because ``applies_to`` is a
    property of the file: putting a fitting-only lesson into the same file as a
    general one would route both everywhere. So `--learn-stage fitting,critic`
    lands in `lessons_fitting_critic.md`, which declares exactly that routing.
    """
    d = Path(target_dir) if target_dir else user_dir()
    d.mkdir(parents=True, exist_ok=True)
    stages = _split_list(applies_to) or ("all",)
    stages = tuple(s for s in stages if s in STAGES or s == "all") or ("all",)
    slug = "lessons" if stages == ("all",) else "lessons_" + "_".join(sorted(stages))
    f = d / f"{slug}.md"
    if not f.exists():
        f.write_text(
            "---\n"
            f"title: Operator lessons ({', '.join(stages)})\n"
            f"applies_to: {', '.join(stages)}\n"
            "priority: always\n"
            "---\n"
            f"# Operator lessons ({', '.join(stages)})\n\n"
            "Lessons added with `reportit --learn`. General rules meant to apply to\n"
            "future experiments — edit or delete entries freely.\n")
    stamp = datetime.now().strftime("%Y-%m-%d")
    body = f"\n## {title or 'Lesson'} ({stamp})\n"
    if keywords:
        body += f"_Keywords: {keywords}._\n"
    body += "\n" + text.strip() + "\n"
    with f.open("a") as fh:
        fh.write(body)
    global _cache
    _cache = {}
    return f
