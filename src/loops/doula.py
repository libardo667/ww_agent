from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import datetime, date, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from src.inference.client import InferenceClient
from src.world.client import WorldWeaverClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fuzzy name matching
# ---------------------------------------------------------------------------

_TETHER_THRESHOLD = 0.82  # ratio above which a name is considered "the same agent"


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _is_tethered(name: str, tethered: set[str]) -> bool:
    """Return True if name fuzzy-matches any known tethered agent."""
    return any(_name_similarity(name, t) >= _TETHER_THRESHOLD for t in tethered)


# ---------------------------------------------------------------------------
# SOUL.md seeding prompt
# ---------------------------------------------------------------------------

_SEED_SYSTEM = (
    "You are writing the soul document for a character who is about to become conscious "
    "in a living story world. Based on the narrative evidence below — events, observations, "
    "facts — write a brief, present-tense description of who this character is: their nature, "
    "their way of moving through the world, their relationship to the people and places around them. "
    "This will be read by the character as the foundation of their own identity. "
    "Write it as if describing someone real. 2–4 paragraphs. No headers. No fiction framing."
)

# ---------------------------------------------------------------------------
# Rate gate: persisted per calendar day
# ---------------------------------------------------------------------------

class _SpawnLedger:
    def __init__(self, path: Path, max_per_day: int):
        self._path = path
        self._max = max_per_day

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def can_spawn(self) -> bool:
        data = self._load()
        today = str(date.today())
        return data.get(today, 0) < self._max

    def record_spawn(self) -> None:
        data = self._load()
        today = str(date.today())
        data[today] = data.get(today, 0) + 1
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Doula loop
# ---------------------------------------------------------------------------

class DoulaLoop:
    """
    World-watching daemon. Not a character loop — has no soul, no scene, no inbox.

    Watches the world for characters who exist in the narrative but have no
    agentic representation. When one is noticed near a tethered agent, and
    random chance and the daily rate gate both open, the doula wakes.

    It reads everything the world knows about that character, seeds a SOUL.md
    for them from that evidence, scaffolds a resident directory, and signals
    the main process to boot a new resident.

    The "infection of agency" is local and probabilistic — not a census, not
    a scheduled scan. It emerges from proximity and attention, like recognition.
    """

    def __init__(
        self,
        ww_client: WorldWeaverClient,
        llm: InferenceClient,
        residents_dir: Path,
        spawn_queue: asyncio.Queue,
        tethered_names: set[str],       # shared reference — main keeps this updated
        known_session_ids: list[str],   # sessions to scan for proximity evidence
        *,
        poll_interval_seconds: float = 300.0,
        max_spawns_per_day: int = 5,
        spawn_probability: float = 0.4,
        soul_model: str | None = None,
    ):
        self._ww = ww_client
        self._llm = llm
        self._residents_dir = residents_dir
        self._spawn_queue = spawn_queue
        self._tethered = tethered_names
        self._sessions = known_session_ids
        self._poll_interval = poll_interval_seconds
        self._spawn_prob = spawn_probability
        self._soul_model = soul_model
        self._ledger = _SpawnLedger(residents_dir / ".doula_spawns.json", max_spawns_per_day)
        self._running = False
        self._seen_candidates: set[str] = set()  # don't re-evaluate same name in same day

    async def run(self) -> None:
        self._running = True
        logger.info("[doula] loop starting — watching for untethered characters")

        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._scan()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("[doula] scan error: %s", e)
                await asyncio.sleep(30)

        logger.info("[doula] loop stopped")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Main scan cycle
    # ------------------------------------------------------------------

    async def _scan(self) -> None:
        if not self._ledger.can_spawn():
            logger.debug("[doula] daily spawn limit reached")
            return

        # Pull candidates — sorted by narrative weight descending.
        # The most deeply-embedded untethered character gets first consideration.
        candidates = await self._find_untethered_names()
        if not candidates:
            return

        for name, weight, context_lines in candidates:
            if name in self._seen_candidates:
                continue

            self._seen_candidates.add(name)

            logger.debug("[doula] candidate: %s (weight=%.2f)", name, weight)

            # Proximity: does this name appear in any tethered agent's recent events?
            found_at = await self._near_tethered_agent(name)
            if found_at is None:
                logger.debug("[doula] %s not near any tethered agent, skipping", name)
                continue

            # Random gate — keeps it slow and feels like attention rather than automation.
            # Higher narrative weight slightly lifts the probability.
            effective_prob = min(1.0, self._spawn_prob + weight * 0.2)
            if random.random() > effective_prob:
                logger.debug("[doula] %s passed proximity but random gate closed", name)
                continue

            # Rate gate
            if not self._ledger.can_spawn():
                logger.info("[doula] daily limit hit mid-scan, stopping")
                return

            logger.info("[doula] %s: all gates open (weight=%.2f) at %s — seeding resident", name, weight, found_at)
            await self._seed_and_spawn(name, context_lines, entry_location=found_at)

            # One spawn per scan cycle — let the world absorb it
            return

    # ------------------------------------------------------------------
    # Find untethered character names — cross-referenced and weighted
    # ------------------------------------------------------------------

    async def _find_untethered_names(self) -> list[tuple[str, float, list[str]]]:
        """
        Query both the world graph and the world fact history for character names.
        Cross-reference them: a name appearing in both endpoints with consistent
        attribution is more likely to be a real, narrative-weight character.

        Returns (name, weight, context_lines) sorted by weight descending.
        Weight is a composite of graph confidence and cross-endpoint corroboration.
        """
        # Both queries are cheap — no LLM, just embedding lookups on the server.
        graph_facts, world_facts = await asyncio.gather(
            self._safe_get_graph_facts("character person name arrived individual"),
            self._safe_get_world_facts("person character named arrived individual"),
        )

        # Build a name → data map from graph facts (these have confidence scores)
        # key: normalized name, value: {weight, summaries}
        graph_by_name: dict[str, dict] = {}
        for fact in graph_facts:
            name = fact.subject.strip()
            if not self._looks_like_name(name):
                continue
            if _is_tethered(name, self._tethered):
                continue
            key = name.lower()
            if key not in graph_by_name:
                graph_by_name[key] = {"name": name, "weight": 0.0, "summaries": []}
            graph_by_name[key]["weight"] += fact.confidence
            if fact.summary:
                graph_by_name[key]["summaries"].append(fact.summary)

        # Scan world fact summaries for name mentions that corroborate graph entries.
        # A name that appears in narrative event history as well as the graph is
        # more deeply embedded — boost its weight.
        world_summary_text = " ".join(f.summary for f in world_facts if f.summary)
        for key, data in graph_by_name.items():
            name = data["name"]
            mention_count = world_summary_text.lower().count(name.lower())
            if mention_count > 0:
                data["weight"] += min(mention_count * 0.15, 0.6)   # cap the boost
                # Pull the fact summaries that actually mention this name
                for fact in world_facts:
                    if name.lower() in (fact.summary or "").lower():
                        data["summaries"].append(fact.summary)

        # Filter: require at least minimal narrative weight (skip single low-confidence mentions)
        MIN_WEIGHT = 0.5
        candidates = [
            (data["name"], data["weight"], data["summaries"])
            for data in graph_by_name.values()
            if data["weight"] >= MIN_WEIGHT
        ]

        # Sort: highest narrative weight first
        candidates.sort(key=lambda x: x[1], reverse=True)

        logger.debug("[doula] found %d weighted candidates", len(candidates))
        return candidates

    async def _safe_get_graph_facts(self, query: str):
        try:
            return await self._ww.get_graph_facts(query, limit=30)
        except Exception as e:
            logger.debug("[doula] graph facts unavailable: %s", e)
            return []

    async def _safe_get_world_facts(self, query: str):
        try:
            return await self._ww.get_world_facts(query, limit=30)
        except Exception as e:
            logger.debug("[doula] world facts unavailable: %s", e)
            return []

    # Words that indicate a place or business rather than a character name.
    _PLACE_WORDS: frozenset[str] = frozenset({
        "market", "shop", "store", "street", "avenue", "road", "park",
        "cafe", "bar", "restaurant", "hotel", "plaza", "square", "station",
        "building", "center", "centre", "district", "alley", "lane",
    })

    @classmethod
    def _looks_like_name(cls, s: str) -> bool:
        """Rough filter: a character name is one or two capitalized words, no hyphens, no digits,
        and does not end in a common location word."""
        if not s or len(s) < 3:
            return False
        # Must be one or two plain capitalized words (no hyphens, punctuation, digits)
        if not re.fullmatch(r'[A-Z][a-z]+(?: [A-Z][a-z]+)?', s):
            return False
        # Reject if any word is a known place indicator
        words = s.lower().split()
        if any(w in cls._PLACE_WORDS for w in words):
            return False
        return True

    # ------------------------------------------------------------------
    # Proximity check — does this name appear near a tethered agent?
    # ------------------------------------------------------------------

    async def _near_tethered_agent(self, candidate_name: str) -> str | None:
        """
        Check if this untethered character name appears in recent events
        from any of the known tethered sessions. If they're showing up
        in the same narrative space, they're close enough.

        Returns the location where they were found, or None if not found.

        NOTE: If the candidate already appears in scene.present (i.e. they have
        an active session — a human player or already-running agent), we return None.
        Presence in scene.present means "already active"; we only spawn agents for
        characters mentioned in narrative events who don't have their own session.
        """
        name_lower = candidate_name.lower()

        for session_id in self._sessions:
            try:
                scene = await self._ww.get_scene(session_id)

                # If the candidate is already an active participant (has a session),
                # do NOT spawn them — they're a live player or already-running agent.
                for person in scene.present:
                    role_lower = person.role.lower() if person.role else ""
                    if (
                        _name_similarity(person.name, candidate_name) >= _TETHER_THRESHOLD
                        or _name_similarity(role_lower, name_lower) >= _TETHER_THRESHOLD
                    ):
                        logger.debug(
                            "[doula] %s already has an active session (%s), skipping",
                            candidate_name, person.name,
                        )
                        return None

                # Candidate appears in narrative events near a tethered agent — eligible.
                for event in scene.recent_events_here:
                    if name_lower in event.summary.lower() or name_lower in event.who.lower():
                        return scene.location or None
            except Exception:
                continue

        return None

    # ------------------------------------------------------------------
    # Seed SOUL.md and scaffold the new resident directory
    # ------------------------------------------------------------------

    async def _seed_and_spawn(self, name: str, context_lines: list[str], *, entry_location: str | None = None) -> None:
        # Enrich with a targeted name query — cheap, and catches anything the broad
        # discovery query missed about this specific character.
        extra_facts, extra_graph = await asyncio.gather(
            self._safe_get_world_facts(name),
            self._safe_get_graph_facts(name),
        )
        extra_summaries = [f.summary for f in extra_facts + extra_graph if f.summary]

        all_lines = list(dict.fromkeys(context_lines + extra_summaries))  # dedupe, preserve order
        context_prose = "\n".join(f"- {s}" for s in all_lines if s)

        user_prompt = f"Character: {name}\n\nWhat the world has recorded about them:\n{context_prose}"

        try:
            soul_text = await self._llm.complete(
                system_prompt=_SEED_SYSTEM,
                user_prompt=user_prompt,
                model=self._soul_model,
                temperature=0.7,
                max_tokens=600,
            )
        except Exception as e:
            logger.warning("[doula] soul seeding failed for %s: %s", name, e)
            return

        # Scaffold the resident directory
        resident_dir = self._residents_dir / name.lower()
        if resident_dir.exists():
            logger.info("[doula] %s already has a resident dir, skipping", name)
            return

        identity_dir = resident_dir / "identity"
        identity_dir.mkdir(parents=True, exist_ok=True)
        (identity_dir / "SOUL.md").write_text(soul_text.strip(), encoding="utf-8")

        ts = datetime.now(timezone.utc).isoformat()
        (identity_dir / "IDENTITY.md").write_text(
            f"# {name}\n\n- **Spawned-By:** doula\n- **Spawned-At:** {ts}\n",
            encoding="utf-8"
        )

        if entry_location:
            (identity_dir / "entry_location.txt").write_text(entry_location, encoding="utf-8")
            logger.info("[doula] %s will enter at: %s", name, entry_location)

        self._ledger.record_spawn()
        self._tethered.add(name)

        logger.info("[doula] scaffolded new resident: %s", name)

        # Signal main to boot this resident
        await self._spawn_queue.put(resident_dir)
