from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ResearchQueue:
    """
    Persistent queue of things the agent wants to look up.

    The slow loop appends items (via RESEARCH: tag extraction from reflections).
    The ground loop pops one item per cycle, fetches a result, and writes it
    to working memory.

    Priority order: high → normal → low. Within each tier, oldest first (FIFO).
    Queue is capped at MAX_ITEMS; when full, the lowest-priority oldest item
    is dropped to make room.
    """

    MAX_ITEMS = 10
    PRIORITIES = ["high", "normal", "low"]

    def __init__(self, path: Path) -> None:
        self._path = path

    def add(self, query: str, priority: str = "normal", source: str = "") -> None:
        """Append a query. Silently drops if query is already queued."""
        query = query.strip()
        if not query:
            return
        items = self._load()
        # Dedup by query text
        if any(i["query"].lower() == query.lower() for i in items):
            return
        items.append({
            "query": query,
            "priority": priority if priority in self.PRIORITIES else "normal",
            "source": source,
            "added_ts": datetime.now(timezone.utc).isoformat(),
        })
        # Trim: if over cap, drop oldest lowest-priority item
        if len(items) > self.MAX_ITEMS:
            for tier in reversed(self.PRIORITIES):
                low_items = [i for i in items if i["priority"] == tier]
                if low_items:
                    items.remove(low_items[0])
                    break
        self._save(items)

    def pop_next(self) -> dict | None:
        """Remove and return the highest-priority oldest item, or None if empty."""
        items = self._load()
        if not items:
            return None
        for tier in self.PRIORITIES:
            tier_items = [i for i in items if i["priority"] == tier]
            if tier_items:
                chosen = tier_items[0]  # oldest in this tier
                items.remove(chosen)
                self._save(items)
                return chosen
        return None

    def __len__(self) -> int:
        return len(self._load())

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, items: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
        )
