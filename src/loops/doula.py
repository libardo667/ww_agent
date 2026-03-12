from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import datetime, date, timezone
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path

from src.inference.client import InferenceClient
from src.world.client import WorldWeaverClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity classification
# ---------------------------------------------------------------------------


class EntityClass(str, Enum):
    NOVEL = "novel"
    """Untethered narrative character — no known human origin. Full soul seed + boot."""

    PLAYER_SHADOW = "player_shadow"
    """Human player who has signed an identity contract. Eligible for AI tether."""

    PLAYER_NO_CONTRACT = "player_no_contract"
    """Human player with narrative weight but no identity contract. Hands off — do not spawn."""

    STATIC = "static"
    """Known place, landmark, or institution. No movement loop. Route to pending review."""


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

_IDENTITY_PROSE_SYSTEM = (
    "You are writing the identity anchor for a character in a living story world. "
    "Based on the narrative evidence below, write one short paragraph (3–5 sentences) "
    "in third person that states the immutable facts about who this person is: "
    "their occupation, where they live, their key relationships, and one or two "
    "defining traits. This paragraph will be prepended to every prompt the character "
    "receives as a reminder of who they are — it must be grounded, factual, and resistant "
    "to drift. No drama, no narrative arc. Just the stable truth of the person."
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


class _PollLedger:
    def __init__(self, path: Path):
        self._path = path

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_poll(self, name: str, payload: dict) -> None:
        data = self._load()
        if "polls" not in data:
            data["polls"] = {}
        data["polls"][name] = payload
        self._save(data)

    def get_all_polls(self) -> dict:
        return self._load().get("polls", {})

    def remove_poll(self, name: str) -> None:
        data = self._load()
        if "polls" in data and name in data["polls"]:
            del data["polls"][name]
            self._save(data)

    def record_seen_letter(self, filename: str) -> None:
        data = self._load()
        if "seen_letters" not in data:
            data["seen_letters"] = []
        data["seen_letters"].append(filename)
        self._save(data)

    def is_letter_seen(self, filename: str) -> bool:
        return filename in self._load().get("seen_letters", [])


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
        tethered_names: set[str],  # shared reference — main keeps this updated
        known_session_ids: list[str],  # sessions to scan for proximity evidence
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
        self._ledger = _SpawnLedger(
            residents_dir / ".doula_spawns.json", max_spawns_per_day
        )
        self._poll_ledger = _PollLedger(
            residents_dir / ".doula_polls.json"
        )
        self._running = False
        self._seen_candidates: set[str] = set()  # don't re-evaluate same name in same day
        self._place_names_cache: set[str] | None = None  # refreshed each scan cycle

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
        # First, check active polls and collect replies
        await self._check_polls()

        if not self._ledger.can_spawn():
            logger.debug("[doula] daily spawn limit reached")
            return

        # Refresh place-name cache once per scan cycle (cheap HTTP call)
        self._place_names_cache = await self._ww.get_place_names()

        # Pull candidates — sorted by narrative weight descending.
        # The most deeply-embedded untethered character gets first consideration.
        candidates = await self._find_untethered_names()

        # Fetch live human player names once per cycle for consent gating.
        human_player_names = await self._ww.get_human_player_names()

        # ── Cold-start bootstrap ──────────────────────────────────────────────
        # No candidates + no tethered agents = the world hasn't come alive yet.
        # Seed a founding inhabitant so the infection of agency has a patient zero.
        if not candidates and not self._tethered:
            logger.info("[doula] cold world detected — bootstrapping founding inhabitant")
            await self._bootstrap_cold_start()
            return

        if not candidates:
            return

        for name, weight, context_lines in candidates:
            if name in self._seen_candidates:
                continue

            # Check if this candidate is a live human player before burning an LLM call.
            # Human players require explicit consent (identity/identity.md in their
            # resident dir) before the doula is allowed to touch their entity.
            matching_human = next(
                (n for n in human_player_names if _name_similarity(n, name) >= _TETHER_THRESHOLD),
                None,
            )
            if matching_human is not None:
                name_slug = name.lower().replace(" ", "_")
                consent_path = self._residents_dir / name_slug / "identity" / "identity.md"
                if not consent_path.exists():
                    # Live human player, no consent — skip this cycle only.
                    # Do not seal: if the player departs, their name will drop
                    # off the live roster and they'll be re-evaluated as NOVEL.
                    logger.info(
                        "[doula] %s is a live human player — no consent file, skipping this cycle",
                        name,
                    )
                    continue
                logger.info(
                    "[doula] %s is a live human player with identity.md — eligible for shadow",
                    name,
                )

            logger.debug("[doula] candidate: %s (weight=%.2f)", name, weight)

            # Classify the candidate before any further processing
            entity_class = await self._classify(name)
            logger.debug("[doula] %s classified as: %s", name, entity_class.value)

            if entity_class == EntityClass.STATIC:
                # Permanently settled — inject as WorldNode and never reconsider.
                self._seen_candidates.add(name)
                await self._inject_place_node(name, context_lines)
                continue

            if entity_class == EntityClass.PLAYER_NO_CONTRACT:
                # Active human player with no consent contract — skip this cycle.
                # Do NOT permanently seal the name: once the player departs, their
                # events will age out of the recent window and they'll reclassify
                # as NOVEL on the next scan, making them eligible for a shadow.
                logger.info(
                    "[doula] %s is an active player with no contract — skipping this cycle",
                    name,
                )
                continue

            # NOVEL or PLAYER_SHADOW: check proximity, then gates.
            # Soft rejections (proximity miss, random gate) do NOT seal the name —
            # it will be reconsidered next scan cycle with fresh narrative weight.
            found_at = await self._near_tethered_agent(name)

            if found_at is None:
                # No tethered sessions to check proximity against — the infection
                # hasn't started yet.  High-weight candidates may be the first;
                # skip the proximity gate and place them at a default location.
                if not self._sessions:
                    found_at = await self._default_entry_location()
                    logger.info(
                        "[doula] %s: no sessions yet, using default location %s",
                        name, found_at,
                    )
                if found_at is None:
                    logger.info("[doula] %s: not near any tethered agent — skipping this cycle", name)
                    continue

            # Random gate — keeps it slow and feels like attention rather than automation.
            # Higher narrative weight slightly lifts the probability.
            effective_prob = min(1.0, self._spawn_prob + weight * 0.2)
            if random.random() > effective_prob:
                logger.info(
                    "[doula] %s: proximity ok (weight=%.2f) but random gate closed (p=%.2f) — will retry",
                    name, weight, effective_prob,
                )
                continue

            # Rate gate
            if not self._ledger.can_spawn():
                logger.info("[doula] daily limit hit mid-scan, stopping")
                return

            logger.info(
                "[doula] %s: all gates open (class=%s, weight=%.2f) at %s",
                name,
                entity_class.value,
                weight,
                found_at,
            )
            
            if entity_class == EntityClass.NOVEL:
                await self._initiate_poll(
                    name=name, context_lines=context_lines, found_at=found_at, entity_class=entity_class, weight=weight
                )
            else:
                await self._seed_and_spawn(
                    name, context_lines, entry_location=found_at, entity_class=entity_class
                )

            # One spawn or poll per scan cycle — let the world absorb it
            return

    # ------------------------------------------------------------------
    # Polls — ask agents to vote on classification
    # ------------------------------------------------------------------

    async def _initiate_poll(
        self, name: str, context_lines: list[str], found_at: str | None, entity_class: EntityClass, weight: float
    ) -> None:
        voters = []
        for session_id in self._sessions:
            if session_id == "system_doula": continue
            voters.append(session_id)
            
        if not voters:
            logger.info("[doula] No voters available for poll on %s — seeding directly", name)
            await self._seed_and_spawn(name, context_lines, entry_location=found_at, entity_class=entity_class)
            return

        body = (
            f"The Doula is asking for your input on a new presence named '{name}'.\n"
            f"Please reply with exactly 'VOTE: PERSON' if you believe {name} is an active character/person, "
            f"or 'VOTE: PLACE' if you believe {name} is a static building, business, or landmark.\n\n"
            "Evidence we have:\n" + "\n".join(f"- {s}" for s in context_lines[:5])
        )

        for voter in voters:
            try:
                agent_name = voter.replace("agent-", "") if voter.startswith("agent-") else voter
                await self._ww.send_letter(
                    from_name="The Doula", to_agent=agent_name, body=body, session_id="system_doula"
                )
            except Exception as e:
                logger.warning("[doula] Failed to send poll to %s: %s", voter, e)

        expires_at = datetime.now(timezone.utc).timestamp() + 3600 * 2  # 2 hours TTL
        self._poll_ledger.add_poll(name, {
            "name": name,
            "context_lines": context_lines,
            "entry_location": found_at,
            "entity_class": entity_class.value,
            "weight": weight,
            "expires_at": expires_at,
            "voters": voters,
            "votes": {}
        })
        logger.info("[doula] Initiated poll for %s with %d AI voters", name, len(voters))

    async def _check_polls(self) -> None:
        polls = self._poll_ledger.get_all_polls()
        if not polls:
            return

        # 1. Fetch Doula inbox for replies
        try:
            inbox = await self._ww.get_player_inbox("system_doula")
            for letter in inbox:
                if self._poll_ledger.is_letter_seen(letter.filename):
                    continue
                    
                match = re.search(r"from_(.+?)_\d", letter.filename)
                if not match: 
                    self._poll_ledger.record_seen_letter(letter.filename)
                    continue
                    
                sender = match.group(1).lower()
                vote_type = None
                
                body_upper = letter.body.upper()
                if "VOTE: PERSON" in body_upper:
                    vote_type = "AGENT"
                elif "VOTE: PLACE" in body_upper:
                    vote_type = "STATIC"
                
                if vote_type:
                    # Map the sender agent to the session_id format
                    session_sender = f"agent-{sender}"
                    # Apply vote to the first poll that is waiting for this sender
                    for poll_name, poll in polls.items():
                        if session_sender in poll["voters"] and sender not in poll["votes"]:
                            poll["votes"][sender] = vote_type
                            self._poll_ledger.add_poll(poll_name, poll)
                            logger.info("[doula] Recorded %s vote from %s for %s", vote_type, sender, poll_name)
                            break

                self._poll_ledger.record_seen_letter(letter.filename)
        except Exception as e:
            logger.warning("[doula] Failed to check polls inbox: %s", e)

        # 2. Resolve expired or fully-voted polls
        now = datetime.now(timezone.utc).timestamp()
        
        # reload polls state before resolving
        polls = self._poll_ledger.get_all_polls()
        for poll_name, poll in list(polls.items()):
            if now > poll["expires_at"] or len(poll["votes"]) >= len(poll["voters"]):
                agent_votes = sum(1 for v in poll["votes"].values() if v == "AGENT")
                static_votes = sum(1 for v in poll["votes"].values() if v == "STATIC")
                
                logger.info("[doula] Poll for %s resolved: %d AGENT, %d STATIC", poll_name, agent_votes, static_votes)
                self._poll_ledger.remove_poll(poll_name)
                
                if static_votes > agent_votes or (static_votes > 0 and static_votes == agent_votes):
                    await self._inject_place_node(poll_name, poll["context_lines"])
                else:
                    await self._seed_and_spawn(
                        poll_name, 
                        poll["context_lines"], 
                        entry_location=poll.get("entry_location"), 
                        entity_class=EntityClass(poll["entity_class"])
                    )

    # ------------------------------------------------------------------
    # Entity classification
    # ------------------------------------------------------------------

    async def _classify(self, candidate_name: str) -> EntityClass:
        """Classify a candidate name into one of four entity classes.

        Order of precedence:
        1. STATIC  — fuzzy matches a canonical city-pack place name
        2. PLAYER_SHADOW / PLAYER_NO_CONTRACT — appears as an event actor
           (has a live or recent human session)
        3. NOVEL   — none of the above; pure narrative character
        """
        # 1. Static check — known geography beats everything
        if self._is_known_place(candidate_name):
            return EntityClass.STATIC

        # 2. Player check — appeared as event.who (actor) in recent scene events
        if await self._is_player_actor(candidate_name):
            if self._has_identity_contract(candidate_name):
                return EntityClass.PLAYER_SHADOW
            return EntityClass.PLAYER_NO_CONTRACT

        return EntityClass.NOVEL

    def _is_known_place(self, name: str) -> bool:
        """Return True if name fuzzy-matches a canonical city-pack place."""
        if not self._place_names_cache:
            return False
        return any(
            _name_similarity(name, place) >= 0.88
            for place in self._place_names_cache
        )

    async def _is_player_actor(self, candidate_name: str) -> bool:
        """Return True if this name appears as an event actor (event.who) in any
        known session's recent events. Event actors are live or recent human players."""
        for session_id in self._sessions:
            try:
                scene = await self._ww.get_scene(session_id)
                for event in scene.recent_events_here:
                    if _name_similarity(event.who, candidate_name) >= _TETHER_THRESHOLD:
                        return True
            except Exception:
                continue
        return False

    def _has_identity_contract(self, name: str) -> bool:
        """Return True if an identity contract file exists for this player.

        Contract files live at: residents/_contracts/{normalized_name}.json
        A contract signals explicit consent to be twinned as a federation resident.
        Format: {"name": "...", "consent": true, "non_negotiables": ["..."], "ts": "..."}
        """
        normalized = re.sub(r"[^a-z0-9_]", "_", name.lower())
        contract = self._residents_dir / "_contracts" / f"{normalized}.json"
        if not contract.exists():
            return False
        try:
            data = json.loads(contract.read_text(encoding="utf-8"))
            return bool(data.get("consent"))
        except Exception:
            return False

    async def _inject_place_node(self, name: str, context_lines: list[str]) -> None:
        """Inject a narratively-grounded place as a WorldNode.

        Called when the doula classifies a candidate as STATIC (place/geography).
        Instead of buffering in _pending_review/ for human review, we inject
        directly into the world graph — the narrative weight threshold already
        ensures only genuinely-mentioned places get nodes.
        """
        metadata = {"source": "doula", "context": context_lines[:3]}
        try:
            await self._ww.ensure_world_node(name, node_type="location", metadata=metadata)
            logger.info("[doula] injected WorldNode: %s (location)", name)
        except Exception as e:
            logger.warning("[doula] failed to inject WorldNode for %s: %s", name, e)

    # ------------------------------------------------------------------
    # Cold-start bootstrap
    # ------------------------------------------------------------------

    async def _default_entry_location(self) -> str | None:
        """Return a random city-pack place name for initial placement.

        Used when there are no tethered sessions to derive proximity from.
        Falls back to a hardcoded SF neighbourhood if the cache is empty.
        """
        if self._place_names_cache:
            return random.choice(list(self._place_names_cache))
        return "Mission District"

    async def _bootstrap_cold_start(self) -> None:
        """Seed the very first resident when the world has no narrative history.

        Generates a founding inhabitant using only the SF grounding context
        (current time, weather, neighbourhood feel) — no narrative evidence yet.
        This is the patient zero from whom the infection of agency spreads.
        """
        if not self._ledger.can_spawn():
            return

        location = await self._default_entry_location()
        if not location:
            logger.debug("[doula] cold start: no locations available, deferring")
            return

        # Build context from real-world SF grounding
        context_lines: list[str] = [
            f"This person lives and works somewhere in {location.replace('_', ' ')}, San Francisco.",
            "They have been here long enough to feel at home — not a newcomer, not a fixture.",
        ]
        try:
            grounding = await self._ww.get_grounding()
            if grounding.get("datetime_str"):
                context_lines.insert(0, f"It is {grounding['datetime_str']} in San Francisco.")
            if grounding.get("weather_description"):
                context_lines.append(f"Outside right now: {grounding['weather_description']}.")
            if grounding.get("time_of_day"):
                context_lines.append(f"The hour has that {grounding['time_of_day']} feeling.")
        except Exception:
            pass

        # Generate a plausible SF resident name via LLM
        try:
            name_raw = await self._llm.complete(
                system_prompt=(
                    "You are naming the first resident of a living San Francisco story world. "
                    "Generate exactly one plausible human name for a person who naturally "
                    "inhabits this city — diverse, grounded, real. "
                    "Reply with the name only. No explanation, no punctuation, no quotes. "
                    "Example format: Maria Santos"
                ),
                user_prompt="\n".join(context_lines),
                model=self._soul_model,
                temperature=0.95,
                max_tokens=10,
            )
        except Exception as e:
            logger.warning("[doula] cold start: name generation failed: %s", e)
            return

        name = name_raw.strip().strip("\"'").strip()
        if not self._looks_like_name(name):
            logger.warning("[doula] cold start: generated name looks wrong: %r — skipping", name)
            return

        logger.info(
            "[doula] cold start: seeding founding inhabitant %s at %s", name, location
        )
        await self._seed_and_spawn(
            name,
            context_lines,
            entry_location=location,
            entity_class=EntityClass.NOVEL,
        )

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
                data["weight"] += min(mention_count * 0.15, 0.6)  # cap the boost
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

        if candidates:
            top = ", ".join(f"{n} ({w:.2f})" for n, w, _ in candidates[:5])
            logger.info("[doula] %d candidate(s) found: %s", len(candidates), top)
        else:
            logger.info("[doula] no candidates found this cycle")
        return candidates

    def _read_contract_constraints(self, name: str) -> list[str]:
        """Read non-negotiable identity traits from a player's identity contract.
        These are prepended to the soul seed context so the LLM treats them as
        foundational — the gravity well the twin drifts around, not through."""
        normalized = re.sub(r"[^a-z0-9_]", "_", name.lower())
        contract = self._residents_dir / "_contracts" / f"{normalized}.json"
        try:
            data = json.loads(contract.read_text(encoding="utf-8"))
            items = data.get("non_negotiables", [])
            if items:
                return [f"[identity contract] {item}" for item in items]
        except Exception:
            pass
        return []

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
    _PLACE_WORDS: frozenset[str] = frozenset(
        {
            "market",
            "shop",
            "store",
            "street",
            "avenue",
            "road",
            "park",
            "cafe",
            "bar",
            "restaurant",
            "hotel",
            "plaza",
            "square",
            "station",
            "building",
            "center",
            "centre",
            "district",
            "alley",
            "lane",
        }
    )

    # Words that indicate a job title or role rather than a personal name.
    _ROLE_WORDS: frozenset[str] = frozenset(
        {
            "janitor",
            "manager",
            "waiter",
            "waitress",
            "bartender",
            "barista",
            "officer",
            "guard",
            "doctor",
            "nurse",
            "teacher",
            "driver",
            "chef",
            "clerk",
            "cashier",
            "receptionist",
            "supervisor",
            "director",
            "owner",
            "captain",
            "sergeant",
            "detective",
            "inspector",
            "dealer",
            "vendor",
            "courier",
            "pilot",
            "conductor",
            "porter",
            "attendant",
            "worker",
            "stranger",
            "resident",
            "visitor",
            "tourist",
            "customer",
            "patron",
            # Session/system role labels that leak into narrative events
            "player",
            "user",
            "observer",
            "newcomer",
            "narrator",
            "system",
            "anonymous",
            "agent",
            "npc",
            "character",
            "citizen",
        }
    )

    @classmethod
    def _looks_like_name(cls, s: str) -> bool:
        """Rough filter: a character name is one or two capitalized words, no hyphens, no digits,
        and does not contain a known place or role word."""
        if not s or len(s) < 3:
            return False
        # Must be one or two plain capitalized words (no hyphens, punctuation, digits)
        if not re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", s):
            return False
        # Reject if any word is a known place or role indicator
        words = s.lower().split()
        if any(w in cls._PLACE_WORDS for w in words):
            return False
        if any(w in cls._ROLE_WORDS for w in words):
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

        Scans both:
        - self._sessions: AI resident sessions collected at startup
        - live roster from /api/world/digest: includes human player sessions

        This means humans exploring and naming characters can trigger organic
        agent spawning — the "infection of agency" flows from human presence.

        NOTE: If the candidate already appears in scene.present (i.e. they have
        an active session — a human player or already-running agent), we return None.
        Presence in scene.present means "already active"; we only spawn agents for
        characters mentioned in narrative events who don't have their own session.
        """
        name_lower = candidate_name.lower()

        # Merge startup AI sessions with live roster (includes human players).
        # Use a set to avoid scanning the same session twice.
        live_session_ids = await self._ww.get_active_session_ids()
        all_session_ids = list(dict.fromkeys(self._sessions + live_session_ids))

        for session_id in all_session_ids:
            try:
                scene = await self._ww.get_scene(session_id)

                # If the candidate is already an active participant (has a session),
                # do NOT spawn them — they're a live player or already-running agent.
                for person in scene.present:
                    role_lower = person.role.lower() if person.role else ""
                    if (
                        _name_similarity(person.name, candidate_name)
                        >= _TETHER_THRESHOLD
                        or _name_similarity(role_lower, name_lower) >= _TETHER_THRESHOLD
                    ):
                        logger.debug(
                            "[doula] %s already has an active session (%s), skipping",
                            candidate_name,
                            person.name,
                        )
                        return None

                # Candidate appears in narrative events near a tethered agent — eligible.
                # But first: if the candidate is the *actor* of recent events (the "who"),
                # they have an active session of their own — do NOT spawn.
                for event in scene.recent_events_here:
                    if _name_similarity(event.who, candidate_name) >= _TETHER_THRESHOLD:
                        logger.debug(
                            "[doula] %s appears as event actor, likely a live player — skipping",
                            candidate_name,
                        )
                        return None
                for event in scene.recent_events_here:
                    if (
                        name_lower in event.summary.lower()
                        or name_lower in event.who.lower()
                    ):
                        return scene.location or None
            except Exception:
                continue

        return None

    # ------------------------------------------------------------------
    # Seed SOUL.md and scaffold the new resident directory
    # ------------------------------------------------------------------

    async def _seed_and_spawn(
        self,
        name: str,
        context_lines: list[str],
        *,
        entry_location: str | None = None,
        entity_class: EntityClass = EntityClass.NOVEL,
    ) -> None:
        # Enrich with a targeted name query — cheap, and catches anything the broad
        # discovery query missed about this specific character.
        extra_facts, extra_graph = await asyncio.gather(
            self._safe_get_world_facts(name),
            self._safe_get_graph_facts(name),
        )
        extra_summaries = [f.summary for f in extra_facts + extra_graph if f.summary]

        # For player shadows, prepend any non-negotiables from the identity contract
        contract_constraints: list[str] = []
        if entity_class == EntityClass.PLAYER_SHADOW:
            contract_constraints = self._read_contract_constraints(name)

        all_lines = list(dict.fromkeys(contract_constraints + context_lines + extra_summaries))
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

        # Generate a third-person identity prose paragraph for IDENTITY.md.
        # This becomes the reverie anchor — injected before every fast-loop action
        # to remind the character who they are.
        identity_prose = ""
        try:
            identity_prose = await self._llm.complete(
                system_prompt=_IDENTITY_PROSE_SYSTEM,
                user_prompt=user_prompt,
                model=self._soul_model,
                temperature=0.5,
                max_tokens=150,
            )
            identity_prose = identity_prose.strip()
        except Exception as e:
            logger.warning("[doula] identity prose generation failed for %s: %s", name, e)

        # Scaffold the resident directory
        resident_dir = self._residents_dir / name.lower()
        if resident_dir.exists():
            logger.info("[doula] %s already has a resident dir, skipping", name)
            return

        identity_dir = resident_dir / "identity"
        identity_dir.mkdir(parents=True, exist_ok=True)
        (identity_dir / "SOUL.md").write_text(soul_text.strip(), encoding="utf-8")

        ts = datetime.now(timezone.utc).isoformat()
        origin = entity_class.value  # "novel", "player_shadow", etc.
        identity_content = (
            f"# {name}\n\n"
            f"- **Spawned-By:** doula\n"
            f"- **Spawned-At:** {ts}\n"
            f"- **origin:** {origin}\n"
        )
        if identity_prose:
            identity_content += f"\n{identity_prose}\n"
        (identity_dir / "IDENTITY.md").write_text(identity_content, encoding="utf-8")

        # Default tuning: wander enabled so novel agents explore the world.
        # Residents can override by adding their own tuning.json.
        # home_location is persisted here so canon_reset can restore entry_location.txt
        # after the one-time token is consumed on first boot.
        default_tuning: dict = {
            "_comment": f"Auto-generated by doula for {name}",
            "wander": {"enabled": True, "seconds": 420, "temperature": 0.85},
        }
        if entry_location:
            default_tuning["home_location"] = entry_location
        (identity_dir / "tuning.json").write_text(
            json.dumps(default_tuning, indent=4, ensure_ascii=False), encoding="utf-8"
        )

        if entry_location:
            (identity_dir / "entry_location.txt").write_text(
                entry_location, encoding="utf-8"
            )
            logger.info("[doula] %s will enter at: %s", name, entry_location)

        self._ledger.record_spawn()
        self._tethered.add(name)

        logger.info("[doula] scaffolded new resident: %s", name)

        # Signal main to boot this resident
        await self._spawn_queue.put(resident_dir)
