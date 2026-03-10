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

    ## Retrieval Upgrade Path

    v1 (current): keyword/tag matching. Fast, zero dependencies, good enough
    for a small memory corpus.

    v2: TF-IDF over memory content. Better relevance, still no external calls.

    v3: Embedding-based semantic search. Best relevance, requires embedding API
    or local model.

    ## Possible v4: WorldWeaver-native retrieval

    WorldWeaver's server already runs an intent/validate/narrate pipeline with
    semantic understanding of the world state. This might be leverageable for
    memory retrieval — the server knows what things *mean* in world context, not
    just which tags match.

    Possible approaches:
    - A dedicated `/api/world/memory/relevant?context=...` endpoint that takes
      the current scene description and returns relevant memory hooks. The server
      could use its existing world fact graph to surface memories the character
      would plausibly recall given where they are and who's present.
    - Piggybacking on the intent parser: pass the current scene as an "intent"
      and let the server's semantic layer identify which stored facts are active.
    - Using the narration model's context window directly: instead of retrieval,
      give the slow loop access to a server-side "what does {name} know about
      {location}/{character}?" query endpoint.

    This would make retrieval world-aware rather than text-aware — a memory
    about Casper surfaces not because it mentions his name but because the world
    model knows he's present and has prior history with this character.

    Pin: discuss with WorldWeaver server team before implementing v3. May be
    able to skip embeddings entirely and go straight to world-native retrieval.
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
