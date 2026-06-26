"""Simple JSON-file disk cache keyed by a string, with md5 helpers.

Makes reruns fast and deterministic: ONCat catalogs, proposal extraction,
and every LLM/probe response are cached under <out>/.reportit_cache/.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def md5(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
    return h.hexdigest()


class Cache:
    def __init__(self, root: Path, enabled: bool = True, bust: bool = False):
        # bust=True (from --refresh): ignore existing entries (every get() misses)
        # but still WRITE fresh results, so this run recomputes everything and the
        # cache is repopulated for next time.
        self.root = Path(root)
        self.enabled = enabled
        self.bust = bust
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # key is already safe-ish; hash to avoid odd characters / length
        return self.root / f"{md5(key)}.json"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled or self.bust:
            return None
        p = self._path(key)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except Exception as e:  # pragma: no cover - corrupt cache
            logger.warning("Cache read failed for %s: %s", key, e)
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        try:
            p.write_text(json.dumps(value, default=str))
        except Exception as e:  # pragma: no cover
            logger.warning("Cache write failed for %s: %s", key, e)

    def get_text(self, key: str) -> Optional[str]:
        if not self.enabled or self.bust:
            return None
        p = self.root / f"{md5(key)}.txt"
        return p.read_text() if p.is_file() else None

    def set_text(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        (self.root / f"{md5(key)}.txt").write_text(value)
