"""
Disk cache for LLM responses, keyed by a content hash.

LLM inference costs money per token; the same headline shouldn't be re-scored (and
re-paid) on every run. The cache key is built from backend + model + the exact
grounded prompt, so it is point-in-time safe (no future info leaks into the key)
and invalidates automatically if the prompt or model changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResponseCache:
    def __init__(self, path, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path)
        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(parts: list[str]) -> str:
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:40]

    def get(self, parts: list[str]):
        if not self.enabled:
            return None
        f = self.path / f"{self._key(parts)}.json"
        if f.exists():
            self.hits += 1
            try:
                return json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                return None
        self.misses += 1
        return None

    def set(self, parts: list[str], value: dict):
        if not self.enabled:
            return
        f = self.path / f"{self._key(parts)}.json"
        f.write_text(json.dumps(value, default=str))

    def stats(self) -> str:
        return f"cache hits={self.hits} misses={self.misses}"
