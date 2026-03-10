from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class MemoryEntry:
    path: Path
    ts: str
    content: str        # written in the character's voice
    tags: list[str]
    source: str         # "promoted_impression" | "slow_reflection" | "letter"

    @classmethod
    def from_file(cls, path: Path) -> MemoryEntry:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            ts=data["ts"],
            content=data["content"],
            tags=data.get("tags", []),
            source=data.get("source", "unknown"),
        )


class LongTermMemory:
    """
    Curated memories persisted by the slow loop.
    Retrieved via tag matching — v1 implementation, no embeddings needed yet.
    Content is always written in the character's voice for direct use in prompts.
    """

    def __init__(self, memory_dir: Path):
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def store(self, content: str, tags: list[str], source: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        safe_ts = ts.replace(":", "-").replace("+", "Z").split("Z")[0] + "Z"
        path = self._dir / f"mem_{safe_ts}.json"
        data = {
            "ts": ts,
            "content": content,
            "tags": tags,
            "source": source,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def retrieve(self, query_tags: list[str], limit: int = 5) -> list[MemoryEntry]:
        """
        Find the most tag-relevant memories for the current context.
        Scored by number of matching tags. Ties broken by recency (newest first).
        """
        if not query_tags:
            return []

        query_set = {t.lower() for t in query_tags}
        scored: list[tuple[int, MemoryEntry]] = []

        for p in self._dir.glob("mem_*.json"):
            try:
                entry = MemoryEntry.from_file(p)
                match_count = len(query_set & {t.lower() for t in entry.tags})
                if match_count > 0:
                    scored.append((match_count, entry))
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        scored.sort(key=lambda x: (x[0], x[1].ts), reverse=True)
        return [entry for _, entry in scored[:limit]]

    def retrieve_as_prose(self, query_tags: list[str], limit: int = 5) -> str:
        """
        Naturalized context string for slow loop prompts.
        Returns retrieved memories as prose, no metadata exposed.
        """
        entries = self.retrieve(query_tags, limit)
        if not entries:
            return ""
        return "\n".join(e.content for e in entries)

    def all_entries(self) -> list[MemoryEntry]:
        """For maintenance/export only. Never used in loop context."""
        entries = []
        for p in sorted(self._dir.glob("mem_*.json")):
            try:
                entries.append(MemoryEntry.from_file(p))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return entries
