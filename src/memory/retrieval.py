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

    ## v4: WorldWeaver-native retrieval — already built, just needs wiring

    WorldWeaver already uses `openai/text-embedding-3-small` (via OpenRouter)
    to embed world events and facts into a semantic graph. Two HTTP endpoints
    expose this directly:

        GET /api/world/facts?query=<text>&session_id=<id>&limit=<N>
            Semantic search over world event history (WorldEvent.embedding)

        GET /api/world/graph/facts?query=<text>&session_id=<id>&limit=<N>
            Semantic search over active world fact graph (WorldFact.embedding)

    Both embed the query text and rank results by cosine similarity. The server
    also computes a blended world context vector (recent events, weighted by
    permanence) used for storylet selection — the same signal could inform
    which agent memories are currently active.

    **Two distinct retrieval problems:**

    1. World knowledge (shared): use the existing /api/world/* endpoints.
       The agent asks "what does the world know about [current context]?" and
       gets back semantically relevant facts and events. No extra embedding cost
       — the server already embeds everything.

    2. Personal memories (character-specific): the long-term memory files in
       this class. These need their own embedding. Options:
       a. Call the same OpenRouter embedding API directly from the agent
          (model: openai/text-embedding-3-small, add embed_text() to inference/)
       b. Add a thin server endpoint that embeds arbitrary text and returns the
          vector — reuses the server's already-configured embedding client.
       c. Stay with tag matching (v1) for personal memories and use the world
          endpoints for world knowledge. Probably good enough for now.

    **Recommended upgrade path:**
    - Wire up world fact retrieval via WorldWeaverClient (free, already exists)
    - Embed personal memories locally using same OpenRouter model when corpus
      grows beyond ~50 entries and tag matching degrades
    - Skip v2/v3 entirely — jump from v1 tags to OpenRouter embeddings when
      the signal matters enough to justify the cost (~$0.00002 per memory)
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
