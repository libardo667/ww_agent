from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.identity.loader import ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.memory.provisional import ProvisionalScratchpad
from src.memory.retrieval import LongTermMemory
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient, world_facts_to_prose

logger = logging.getLogger(__name__)

# The slow loop has no world action client — capability enforced structurally.
# It can stage letter drafts and note soul shifts. That's the extent of its reach.

# ---- Subconscious pattern matching ----
# We match on the subconscious's natural-language description, not on agent output.

_CONTACT_WORDS = re.compile(
    r'\b(write|letter|reach out|send|tell|say|speak|contact|reply|message|note)\b',
    re.IGNORECASE,
)

_SHIFT_WORDS = re.compile(
    r'\b(shift|shifted|shifted in|changed|change|different|no longer|come to see|'
    r'realize|realized|reckon|reckoning|something has|has changed|who they are|'
    r'their sense|identity|now sees|now feels|settled|unsettled|moved)\b',
    re.IGNORECASE,
)

# Subconscious system prompt — reads reflection cold, describes what it notices.
# It produces natural language, not structured output. The framework reads that NL.
_SUBCONSCIOUS_SYSTEM = (
    "You are reading someone's private journal entry alongside a record of what they've been doing. "
    "Describe, in plain natural language, what this person seems to want to do next — "
    "who (if anyone) they seem to want to reach out to, and whether anything seems to have "
    "shifted in who they are. Be specific but brief. Write as if noting observations to yourself."
)


class SlowLoop(BaseLoop):
    """
    Introspective processing loop. Fires when enough impressions accumulate,
    or as a fallback timer.

    The slow loop does NOT act in the world. It is the character sitting with
    what they've been doing — processing what the fast loop left behind,
    making sense of it, deciding who to write to, noticing what has shifted
    in themselves.

    Architecture: two passes.

    Pass 1 — Reflective: the agent writes completely freely. No format hints,
    no tags, no framework vocabulary. System prompt is SOUL.md verbatim.
    User prompt is their recent history and impressions, in prose.

    Pass 2 — Subconscious: a separate, cheaper LLM call reads the reflection
    cold and describes in plain language what it noticed: any intentions, any
    relationships on their mind, any identity shifts. Natural language output only.
    The framework pattern-matches on this to decide what to do.

    Optionally, a third targeted call drafts the actual letter body if contact
    intention was detected — again, no format instructions, just the agent writing.

    No [ACTION: ...] tag. No world client. No format requirements on agent output.
    """

    def __init__(
        self,
        identity: ResidentIdentity,
        resident_dir: Path,
        ww_client: WorldWeaverClient,
        llm: InferenceClient,
        session_id: str,
        working_memory: WorkingMemory,
        provisional: ProvisionalScratchpad,
        long_term: LongTermMemory,
    ):
        super().__init__(identity.name, resident_dir)
        self._identity = identity
        self._ww = ww_client        # read-only: world facts retrieval only
        self._llm = llm
        self._session_id = session_id
        self._working = working_memory
        self._provisional = provisional
        self._long_term = long_term
        self._tuning = identity.tuning
        self._decisions_dir = resident_dir / "decisions"
        self._decisions_dir.mkdir(parents=True, exist_ok=True)
        self._decision_count = len(list(self._decisions_dir.glob("decision_*.json")))

    # ------------------------------------------------------------------
    # Trigger: impression threshold OR fallback timer
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        fallback = self._tuning.slow_fallback_seconds
        poll_interval = 15.0
        elapsed = 0.0

        while elapsed < fallback:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            pending = self._provisional.pending_impressions()
            if len(pending) >= self._tuning.slow_impression_threshold:
                return

        logger.debug("[%s:slow] fallback timer fired", self.name)

    # ------------------------------------------------------------------
    # Context: what the fast loop has been doing + world memory
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict:
        pending = self._provisional.pending_impressions()
        recent = self._working.all()

        locations = [e.get("location", "") for e in recent[-5:] if isinstance(e, dict)]
        people = []
        for imp in pending:
            people.extend(imp.colocated)
        query_text = " ".join(filter(None, set(locations) | set(people)))

        world_facts = []
        if query_text:
            try:
                world_facts = await self._ww.get_world_facts(query_text, self._session_id, limit=5)
            except Exception as e:
                logger.debug("[%s:slow] world facts unavailable: %s", self.name, e)

        long_term = self._long_term.retrieve(
            list(filter(None, set(locations) | set(people))), limit=5
        )

        return {
            "pending": pending,
            "recent": recent,
            "world_facts": world_facts,
            "long_term": long_term,
        }

    async def _should_act(self, context: dict) -> bool:
        return True

    # ------------------------------------------------------------------
    # Pass 1 — Reflective: agent writes freely, no format hints
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        pending = context["pending"]
        recent = context["recent"]
        world_facts = context["world_facts"]
        long_term = context["long_term"]

        prompt_parts: list[str] = []

        # What the fast loop has been doing — presented as their own recent history
        if recent:
            action_lines = [
                e["action"] for e in recent[-self._tuning.slow_max_context_events:]
                if isinstance(e, dict) and e.get("action")
            ]
            if action_lines:
                prompt_parts.append("What you've been doing:\n" + "\n".join(f"- {a}" for a in action_lines))

        # What the fast loop was noticing — rendered as prose, no file paths or status fields
        impressions_prose = self._provisional.pending_as_prose()
        if impressions_prose:
            prompt_parts.append(impressions_prose)

        # World context surfaced by the places and people in those impressions
        if world_facts:
            facts_prose = world_facts_to_prose(world_facts)
            if facts_prose:
                prompt_parts.append(facts_prose)

        # Personal memories that the current context activates
        if long_term:
            memory_lines = [m.content for m in long_term if m.content]
            if memory_lines:
                prompt_parts.append("\n".join(memory_lines))

        # No format instructions. The agent writes whatever they write.
        user_prompt = "\n\n".join(prompt_parts)

        reflection = await self._llm.complete(
            system_prompt=self._identity.soul,
            user_prompt=user_prompt,
            model=self._tuning.slow_model,
            temperature=self._tuning.slow_temperature,
            max_tokens=self._tuning.slow_max_tokens,
        )

        # ------------------------------------------------------------------
        # Pass 2 — Subconscious: reads the reflection cold, describes what it noticed
        # ------------------------------------------------------------------

        # Build a brief account of recent actions for the subconscious to read alongside
        recent_summary = ""
        if recent:
            action_lines = [
                e["action"] for e in recent[-self._tuning.slow_max_context_events:]
                if isinstance(e, dict) and e.get("action")
            ]
            if action_lines:
                recent_summary = "What they've been doing:\n" + "\n".join(f"- {a}" for a in action_lines) + "\n\n"

        subconscious_user = recent_summary + "Their journal entry:\n\n" + reflection

        subconscious_reading = await self._llm.complete(
            system_prompt=_SUBCONSCIOUS_SYSTEM,
            user_prompt=subconscious_user,
            model=self._tuning.slow_subconscious_model,
            temperature=0.4,
            max_tokens=300,
        )

        logger.debug("[%s:slow] subconscious: %s", self.name, subconscious_reading[:120])

        # ------------------------------------------------------------------
        # Framework interpretation — pattern match on subconscious NL
        # ------------------------------------------------------------------

        await self._interpret_and_act(reflection, subconscious_reading, pending, recent)

    # ------------------------------------------------------------------
    # Interpret the subconscious's NL and act accordingly
    # ------------------------------------------------------------------

    async def _interpret_and_act(
        self,
        reflection: str,
        subconscious_reading: str,
        pending,
        recent: list,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        # Detect contact intention: name + contact-leaning language in subconscious output
        letter_recipient = self._detect_contact_intent(subconscious_reading)

        # Detect identity shift: shift-language in subconscious output
        soul_note = self._detect_identity_shift(subconscious_reading)

        # Stage a letter intent if the subconscious detected contact desire.
        # The mail loop picks this up and asks the agent what they want to say —
        # the letter is written there, not here. The slow loop doesn't draft letters.
        if letter_recipient:
            self._stage_letter_intent(letter_recipient, subconscious_reading)
            logger.info("[%s:slow] staged letter intent for %s", self.name, letter_recipient)

        # Append a soul note if a shift was sensed
        if soul_note:
            self._record_soul_note(soul_note, now)
            logger.info("[%s:slow] soul note: %s", self.name, soul_note)

        # Decision log — records both the reflection and what the subconscious read into it
        self._decision_count += 1
        decision_path = self._decisions_dir / f"decision_{self._decision_count}.json"
        decision_path.write_text(json.dumps({
            "ts": now,
            "loop": "slow",
            "reflection": reflection,
            "subconscious": subconscious_reading,
            "letter_to": letter_recipient,
            "soul_note": soul_note,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # Archive impressions — they've been reflected on
        for imp in pending:
            self._provisional.archive(imp, reflection[:200])

        # Store a long-term memory from this reflection
        if len(reflection) > 50:
            tags = list({
                e.get("location", "") for e in self._working.recent(5)
                if isinstance(e, dict) and e.get("location")
            })
            self._long_term.store(reflection[:400], tags=tags, source="slow_reflection")

    # ------------------------------------------------------------------
    # NL pattern matching on subconscious output
    # ------------------------------------------------------------------

    def _detect_contact_intent(self, subconscious_reading: str) -> str | None:
        """
        Look for a name + contact-intention in the subconscious's natural language.
        Returns the first name detected in that context, or None.
        """
        if not _CONTACT_WORDS.search(subconscious_reading):
            return None

        # Find capitalized names that appear alongside contact words.
        # Scan sentence by sentence — if a sentence has a contact word and a proper name, extract it.
        for sentence in re.split(r'[.!?\n]+', subconscious_reading):
            if _CONTACT_WORDS.search(sentence):
                name_match = re.search(r'\b([A-Z][a-z]{2,})\b', sentence)
                if name_match:
                    candidate = name_match.group(1)
                    # Skip common non-name words that happen to be capitalized mid-sentence
                    if candidate not in {"They", "The", "Their", "This", "There", "That",
                                         "Some", "It", "What", "Who", "How", "When", "Where"}:
                        return candidate

        return None

    def _detect_identity_shift(self, subconscious_reading: str) -> str | None:
        """
        Look for identity-shift language in the subconscious's reading.
        If found, return a brief sentence extracted from its description.
        """
        if not _SHIFT_WORDS.search(subconscious_reading):
            return None

        # Find the sentence most saturated with shift language — use it as the note
        best = None
        best_count = 0
        for sentence in re.split(r'[.!?\n]+', subconscious_reading):
            count = len(_SHIFT_WORDS.findall(sentence))
            if count > best_count and len(sentence.strip()) > 10:
                best = sentence.strip()
                best_count = count

        return best

    def _stage_letter_intent(self, recipient: str, subconscious_reading: str) -> None:
        """
        Stage a minimal intent file for the mail loop to act on.
        The mail loop will ask the agent what they want to say — the letter
        is written there, not here. We only carry enough context for the
        mail loop to frame the question naturally.
        """
        intents_dir = self.resident_dir / "letters" / "intents"
        intents_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        intent_path = intents_dir / f"intent_{ts}_{recipient}.md"

        # Pull a short excerpt from the subconscious reading as context —
        # just enough for the mail loop to ground the question naturally.
        excerpt = subconscious_reading.strip()[:300]

        intent_path.write_text(
            f"To: {recipient}\nStaged-At: {ts}\n\nContext:\n{excerpt}",
            encoding="utf-8"
        )

    def _record_soul_note(self, note: str, ts: str) -> None:
        """
        Append a soul note to SOUL.md rather than rewriting it wholesale.
        The character earns small edits through accumulated experience.
        The operator can periodically integrate notes into the main text.
        """
        soul_path = self.resident_dir / "identity" / "SOUL.md"
        if soul_path.exists():
            current = soul_path.read_text(encoding="utf-8")
            soul_path.write_text(
                current + f"\n\n---\n*{ts[:10]}:* {note}",
                encoding="utf-8"
            )

    async def _cooldown(self) -> None:
        await asyncio.sleep(5.0)
