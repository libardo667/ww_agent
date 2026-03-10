# src/memory/ — Memory System

## Design Philosophy

Humans don't re-read their life history before every decision. They have a
current state shaped by experience, a small working buffer of recent events,
and retrieval-triggered recall for older memories. This memory system mirrors
that architecture.

## Three Memory Layers

### 1. Working Memory (`working.py`)

**What:** A small rolling buffer of the last N world events (default: 20).
**Who reads it:** Slow loop (full buffer), Fast loop (last 5 only).
**Who writes it:** Both loops append after each turn.
**Persistence:** `{resident_dir}/memory/working.json`
**Eviction:** FIFO. When buffer exceeds N, oldest entries drop off.

This is "what just happened." It's cheap to load, small enough to fit in
any context window, and provides the slow loop with enough recent history
to make decisions without scanning the entire world.

```python
class WorkingMemory:
    def __init__(self, path: Path, max_items: int = 20):
        ...
    def append(self, event: dict) -> None:
        """Add an event, evict oldest if over capacity."""
    def recent(self, n: int = 5) -> list[dict]:
        """Return the last N events (for fast loop)."""
    def all(self) -> list[dict]:
        """Return the full buffer (for slow loop)."""
    def save(self) -> None:
        """Persist to disk."""
```

### 2. Provisional Scratchpad (`provisional.py`)

**What:** Timestamped impression fragments written by the fast loop.
**Who reads it:** Slow loop only.
**Who writes it:** Fast loop only.
**Persistence:** `{resident_dir}/memory/provisional/imp_{timestamp}.json`
**Lifecycle:** pending → promoted/archived/discarded by slow loop.

This is the two-pass cognition mechanism. The fast loop blurts a raw
reaction without interpretation. The slow loop reads it later and decides
what it means.

```python
class ProvisionalScratchpad:
    def __init__(self, provisional_dir: Path, archive_dir: Path):
        ...
    def write_impression(self, trigger: str, reaction: str, location: str,
                         colocated: list[str]) -> Path:
        """Fast loop writes a new impression. Returns the file path."""
    def pending_impressions(self) -> list[Impression]:
        """Slow loop reads all pending (unprocessed) impressions."""
    def promote(self, impression: Impression, decision_entry: str) -> None:
        """Slow loop promotes an impression into the decision log."""
    def archive(self, impression: Impression, interpretation: str) -> None:
        """Slow loop archives with a note."""
    def discard(self, impression: Impression) -> None:
        """Slow loop silently discards."""
```

### 3. Long-Term Memory / Retrieval (`retrieval.py`)

**What:** Curated facts, decisions, and relationship notes persisted by the
slow loop over time.
**Who reads it:** Slow loop only, via semantic search triggered by current context.
**Who writes it:** Slow loop only (promotes from provisional, writes reflections).
**Persistence:** `{resident_dir}/memory/long_term/` — markdown files, indexed.

This is NOT a full event log. It's the distilled, curated residue of
experience — like MEMORY.md in OpenClaw, but structured for retrieval.

The slow loop doesn't scan all of long-term memory every cycle. Instead, it
builds a query from the current context (location, colocated characters,
provisional impressions) and retrieves the top-K most relevant memories.

**Implementation options (in order of complexity):**
1. Simple keyword/tag match (v1 — sufficient for MVP)
2. TF-IDF over memory entries (v2 — better relevance)
3. Embedding-based semantic search (v3 — requires embedding calls)

For v1, each long-term memory entry has tags derived from its content.
The slow loop queries by current location + character names + impression
keywords. Good enough to start.

```python
class LongTermMemory:
    def __init__(self, memory_dir: Path):
        ...
    def store(self, content: str, tags: list[str], source: str) -> None:
        """Write a new memory entry."""
    def retrieve(self, query_tags: list[str], limit: int = 5) -> list[MemoryEntry]:
        """Find the most relevant memories for the current context."""
    def all_entries(self) -> list[MemoryEntry]:
        """For maintenance/export only. Never used in loop context."""
```

## Memory and Identity

SOUL.md is NOT memory. It's disposition — the compressed, evolved personality
that survives world resets. Memory is episodic — what happened, when, where.
SOUL.md is temperamental — how I process what happens.

When the slow loop updates SOUL.md, it's not "remembering." It's *changing
who it is* based on accumulated experience. That's character development, not
recall.

When a world resets, memory is wiped but SOUL.md persists. The character
arrives in a new world with the same personality but no episodic history.
This is by design.

## Memory and Prompt Naturalization

All internal memory vocabulary — "pending," "promote," "archive," "discard,"
"working," "provisional," file paths — is framework language. None of it
should ever appear in a prompt that reaches the LLM.

When constructing context for each loop, translate memory data into prose
the character would recognize as their own experience:

- **Working memory** becomes: *"Recently: ..."* followed by natural event
  descriptions, not JSON entries.

- **Provisional impressions** become: *"Lately you've been noticing..."*
  followed by the raw reactions in first person — not a list of impression
  objects with status fields.

- **Long-term memories** become: retrieved passages written in the character's
  voice, surfaced as if naturally recalled — not labeled as "retrieved memory
  entry" or tagged with metadata.

The framework manages the filing. The agent experiences the content.
The distinction between "what I'm currently aware of" and "what I remember
from before" should feel like the difference between what's in front of you
right now and what you're carrying from the past — not like two different
database tables.
