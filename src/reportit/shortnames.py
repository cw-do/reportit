"""Short display names for long reduced-output filenames.

EQSANS output names encode the whole reduction in the filename, e.g.

    merged_leaf1_370_70_4m_2p5A_30Hz_4m2.5a30hz_frame0_4m2.5a30hz_frame1_Iq.txt

Used verbatim in a table column or a plot legend those names overflow the page
and swamp the figure. This module derives a short label for each one and keeps
the full name available for a mapping table, so nothing is lost — the report
states which short name is which file.

The shortening is *derived*, not arbitrary: the tokens every name shares carry no
information for telling them apart, so the common leading and trailing tokens are
dropped and what remains is the part that actually distinguishes the samples:

    merged_leaf1_370_70_4m_2p5A_30Hz_frame0_frame1  ->  leaf1_370_70
    merged_leaf2_dark457_78_4m_2p5A_30Hz_frame0_frame1  ->  leaf2_dark457_78

Only when that still leaves an unwieldy name does it fall back to an indexed
label (S1, S2, ...), which the mapping table then explains.
"""

from __future__ import annotations

import re
from typing import Iterable

# Names at or below this length are already readable — never touched.
TRIGGER_LEN = 15
# A derived name must come in at or below this to be worth using.
MAX_SHORT_LEN = 24
# Only names longer than this justify falling back to opaque indexed labels
# (S1, S2, ...). Between TRIGGER_LEN and here, names are left alone: they fit a
# table well enough, and a legible long name beats an opaque short one.
OVERFLOW_LEN = 30

_SPLIT = re.compile(r"[_]+")


class NameMap:
    """Full display name -> short label. Unknown names pass through unchanged."""

    def __init__(self, mapping: dict[str, str] | None = None,
                 drop_head: list[str] | None = None,
                 drop_tail: list[str] | None = None):
        self._map: dict[str, str] = dict(mapping or {})
        # the shared leading/trailing tokens that were dropped, so labels that
        # are not themselves dataset names (group titles) can be shortened alike
        self._drop_head: list[str] = list(drop_head or [])
        self._drop_tail: list[str] = list(drop_tail or [])

    def short(self, name) -> str:
        if name is None:
            return ""
        return self._map.get(str(name), str(name))

    def update(self, other: "NameMap") -> None:
        for full, sh in other._map.items():
            self._map.setdefault(full, sh)
        if not self._drop_head:
            self._drop_head = list(other._drop_head)
        if not self._drop_tail:
            self._drop_tail = list(other._drop_tail)

    def shorten_label(self, text) -> str:
        """Shorten a label that may not be a dataset name (e.g. a group title)
        by dropping the same shared tokens. Returns it unchanged if that would
        empty it or gains nothing."""
        if text is None:
            return ""
        text = str(text)
        if text in self._map:
            return self._map[text]
        if not (self._drop_head or self._drop_tail):
            return text
        toks = _tokens(text)
        while toks and self._drop_head and toks[0] in self._drop_head and len(toks) > 1:
            toks.pop(0)
        while toks and self._drop_tail and toks[-1] in self._drop_tail and len(toks) > 1:
            toks.pop()
        out = "_".join(toks)
        return out if out and len(out) < len(text) else text

    @property
    def active(self) -> bool:
        """True if anything was actually shortened."""
        return any(full != sh for full, sh in self._map.items())

    def pairs(self) -> list[tuple[str, str]]:
        """(short, full) for names that were shortened, deduped and sorted."""
        seen: dict[str, str] = {}
        for full, sh in self._map.items():
            if full == sh:
                continue
            # if two full names share a short label, keep the longest full name
            if sh not in seen or len(full) > len(seen[sh]):
                seen[sh] = full
        return sorted(seen.items(), key=lambda p: _natkey(p[0]))

    def __len__(self) -> int:
        return len(self._map)


def _natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _tokens(name: str) -> list[str]:
    return [t for t in _SPLIT.split(name) if t]


def _common_prefix_len(token_lists: list[list[str]]) -> int:
    if len(token_lists) < 2:
        return 0
    n = 0
    shortest = min(len(t) for t in token_lists)
    while n < shortest - 1:  # never consume every token
        tok = token_lists[0][n]
        if all(t[n] == tok for t in token_lists):
            n += 1
        else:
            break
    return n


def _common_suffix_len(token_lists: list[list[str]], head: int) -> int:
    if len(token_lists) < 2:
        return 0
    n = 0
    # leave at least one token beyond the common prefix
    shortest = min(len(t) - head for t in token_lists)
    while n < shortest - 1:
        tok = token_lists[0][-(n + 1)]
        if all(t[-(n + 1)] == tok for t in token_lists):
            n += 1
        else:
            break
    return n


def build(names: Iterable[str], *, trigger_len: int = TRIGGER_LEN,
          max_short_len: int = MAX_SHORT_LEN,
          overflow_len: int = OVERFLOW_LEN) -> NameMap:
    """Derive short labels for ``names``.

    All-or-nothing by design: either every name in the collection is shortened
    the same way, or none is. A table mixing derived names with opaque ``S7``
    labels is harder to read than one that does neither.

    1. no name exceeds ``trigger_len``      -> leave everything alone;
    2. dropping the shared leading/trailing tokens gets every name under
       ``max_short_len`` and actually shortens things -> use those;
    3. otherwise, if names still exceed ``overflow_len`` -> indexed labels;
    4. otherwise -> leave everything alone.
    """
    uniq = sorted({str(n) for n in names if n is not None and str(n) != ""})
    if not uniq or not any(len(n) > trigger_len for n in uniq):
        return NameMap({n: n for n in uniq})

    token_lists = [_tokens(n) for n in uniq]
    head = _common_prefix_len(token_lists)
    tail = _common_suffix_len(token_lists, head)

    cands = []
    for name, toks in zip(uniq, token_lists):
        core = toks[head: len(toks) - tail] if tail else toks[head:]
        cands.append("_".join(core) or name)

    derived_ok = (
        len(set(cands)) == len(cands)                 # still distinguishable
        and max(len(c) for c in cands) <= max_short_len
        and max(len(c) for c in cands) < max(len(n) for n in uniq)  # a real gain
    )
    if derived_ok:
        head_toks = token_lists[0][:head] if head else []
        tail_toks = token_lists[0][len(token_lists[0]) - tail:] if tail else []
        return NameMap(dict(zip(uniq, cands)), head_toks, tail_toks)

    if max(len(n) for n in uniq) > overflow_len:
        # Nothing derivable, but the names really are too long for a table.
        return NameMap({n: f"S{i}" for i, n in enumerate(sorted(uniq, key=_natkey), 1)})

    return NameMap({n: n for n in uniq})   # long-ish but legible — leave alone
