from __future__ import annotations

import json
from pathlib import Path


class WorkingMemory:
    """Small rolling buffer of recent world events. FIFO eviction at max_items."""

    def __init__(self, path: Path, max_items: int = 20):
        self._path = path
        self._max_items = max_items
        self._items: list[dict] = []
        if path.exists():
            try:
                self._items = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._items = []

    def append(self, event: dict) -> None:
        self._items.append(event)
        if len(self._items) > self._max_items:
            self._items = self._items[-self._max_items:]
        self.save()

    def recent(self, n: int = 5) -> list[dict]:
        """Last N events — for the fast loop."""
        return self._items[-n:]

    def all(self) -> list[dict]:
        """Full buffer — for the slow loop."""
        return list(self._items)

    def has_any(self) -> bool:
        """True if this resident has acted before (not a first boot)."""
        return len(self._items) > 0

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._items, indent=2, ensure_ascii=False), encoding="utf-8")
