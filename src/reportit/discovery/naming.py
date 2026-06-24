"""Best-effort parsing of EQSANS reduced-output names.

These are *hints* for grouping/labeling, NOT the source of truth — the LLM
strategist makes the real decisions. Convention observed: [BASE]_[TEMP?]_[CONFIG]
e.g. "pb30.i10_20C_2.5m2.5a", "porsil_4m10a", "1_20C_2.5m2.5a".
"""

from __future__ import annotations

import re

# config like 2.5m2.5a, 4m10a, 4m2.5a, optionally a frequency suffix like 30hz
_CONFIG_RE = re.compile(r"(?P<config>\d+(?:\.\d+)?m\d+(?:\.\d+)?a(?:\d+hz)?)$", re.IGNORECASE)
_TEMP_RE = re.compile(r"_(?P<temp>-?\d+(?:\.\d+)?C)(?=_|$)", re.IGNORECASE)

_STANDARD_PREFIXES = ("porsil", "porasil", "blank", "empty")


def parse_sample_name(name: str) -> tuple[str, str | None, str | None]:
    """Return (base, temperature, config). Any field may be None if not present."""
    stem = name
    # strip a trailing _Iq / _Iqxqy if present
    stem = re.sub(r"_(Iqxqy|Iq)$", "", stem, flags=re.IGNORECASE)

    config = None
    m = _CONFIG_RE.search(stem)
    if m:
        config = m.group("config")
        stem = stem[: m.start()].rstrip("_")

    temp = None
    tm = _TEMP_RE.search("_" + stem + "_")
    if tm:
        temp = tm.group("temp")
        stem = re.sub(r"_?" + re.escape(temp) + r"$", "", stem, flags=re.IGNORECASE).rstrip("_")

    base = stem if stem else name
    return base, temp, config


def is_standard(base: str) -> bool:
    b = base.lower()
    return any(b.startswith(p) for p in _STANDARD_PREFIXES)
