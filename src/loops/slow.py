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
from src.memory.research_queue import ResearchQueue
from src.memory.retrieval import LongTermMemory
from src.memory.reveries import ReverieDeck
from src.memory.voice import VoiceDeck
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient, world_facts_to_prose

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Satiation: topics the agent has already reflected on heavily this session.
# Key = normalized topic string (person name or location slug).
# Value = count of slow loop firings that included this topic.
# When a topic exceeds SATIATION_THRESHOLD, impressions dominated by that
# topic are skipped in the next pass to break the feedback spiral.
# ---------------------------------------------------------------------------
SATIATION_THRESHOLD = 3   # reflections on same topic before cooling down
SATIATION_DECAY = 2       # decrement satiation score each firing (to allow re-emergence)

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
        reveries: ReverieDeck,
        voice: VoiceDeck,
        research_queue: ResearchQueue | None = None,
    ):
        super().__init__(identity.name, resident_dir)
        self._identity = identity
        self._ww = ww_client        # read-only: world facts retrieval only
        self._llm = llm
        self._session_id = session_id
        self._working = working_memory
        self._provisional = provisional
        self._long_term = long_term
        self._reveries = reveries
        self._voice = voice
        self._research_queue = research_queue
        self._tuning = identity.tuning
        self._decisions_dir = resident_dir / "decisions"
        self._decisions_dir.mkdir(parents=True, exist_ok=True)
        self._decision_count = len(list(self._decisions_dir.glob("decision_*.json")))
        # Refractory: timestamp of the last slow loop firing.
        # Prevents rapid re-firing even when impressions pile up immediately after.
        self._last_fire_ts: float = 0.0
        # Satiation: per-topic reflection counts. Decremented each firing.
        self._satiation: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Trigger: impression threshold OR fallback timer
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        fallback = self._tuning.slow_fallback_seconds
        # Refractory: minimum gap between slow loop firings.
        # Prevents a fresh batch of impressions from immediately re-triggering
        # after a firing — the core mechanism that breaks narrative spirals.
        refractory_seconds = getattr(self._tuning, "slow_refractory_seconds", 240.0)
        poll_interval = 15.0
        elapsed = 0.0

        while elapsed < fallback:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            # Fast loop introspect signal: fires us early when the lizard brain
            # decides now is a good moment to reflect. Refractory still applies.
            signal_path = self.resident_dir / "memory" / "introspect_signal"
            if signal_path.exists():
                import time
                since_last = time.monotonic() - self._last_fire_ts
                if since_last >= refractory_seconds:
                    try:
                        signal_path.unlink()
                    except OSError:
                        pass
                    logger.info("[%s:slow] introspect signal received — firing early", self.name)
                    return
                else:
                    logger.debug(
                        "[%s:slow] introspect signal ignored — refractory active (%.0fs left)",
                        self.name, refractory_seconds - since_last,
                    )

            pending = self._provisional.pending_impressions()
            if len(pending) >= self._tuning.slow_impression_threshold:
                # Respect refractory period even when threshold is met
                import time
                since_last = time.monotonic() - self._last_fire_ts
                if since_last >= refractory_seconds:
                    return
                else:
                    remaining = refractory_seconds - since_last
                    logger.debug(
                        "[%s:slow] impression threshold met but refractory active (%.0fs left)",
                        self.name, remaining,
                    )

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

        # Apply satiation filter: skip impressions whose topics have been
        # over-represented in recent slow loop firings.
        pending = self._apply_satiation(pending)

        world_facts = []
        if query_text:
            try:
                world_facts = await self._ww.get_world_facts(query_text, self._session_id, limit=5)
            except Exception as e:
                logger.debug("[%s:slow] world facts unavailable: %s", self.name, e)

        long_term = self._long_term.retrieve(
            list(filter(None, set(locations) | set(people))), limit=5
        )

        # Geographic context: ground the agent's reflection in real city geography.
        # Use the most recent location we have a record of.
        map_context = ""
        current_location = locations[-1] if locations else ""
        if current_location:
            try:
                map_context = await self._ww.get_location_map_context(
                    self._session_id, current_location
                )
            except Exception as e:
                logger.debug("[%s:slow] map context unavailable: %s", self.name, e)

        return {
            "pending": pending,
            "recent": recent,
            "world_facts": world_facts,
            "long_term": long_term,
            "map_context": map_context,
        }

    async def _should_act(self, context: dict) -> bool:
        return True

    # ------------------------------------------------------------------
    # Pass 1 — Reflective: agent writes freely, no format hints
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        import time
        self._last_fire_ts = time.monotonic()  # record firing for refractory

        pending = context["pending"]
        recent = context["recent"]
        world_facts = context["world_facts"]
        long_term = context["long_term"]
        map_context: str = context.get("map_context", "")

        # Update satiation counts for topics appearing in this firing
        self._update_satiation(pending)

        prompt_parts: list[str] = []

        # Geographic grounding — city bones before the character's inner world.
        # Presented as contextual fact, not instruction. The character doesn't need
        # to "use" it — it just sits in their awareness the way real knowledge does.
        if map_context:
            prompt_parts.append(map_context)

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
            system_prompt=self._identity.soul_with_context,
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

        # Stage a letter intent if the subconscious detected contact desire.
        # The mail loop picks this up and asks the agent what they want to say —
        # the letter is written there, not here. The slow loop doesn't draft letters.
        if letter_recipient:
            self._stage_letter_intent(letter_recipient, subconscious_reading)
            logger.info("[%s:slow] staged letter intent for %s", self.name, letter_recipient)

        # Detect identity shift: shift-language in subconscious output.
        # If shift is sensed, ask the character to capture it in their own voice —
        # a brief first-person fragment, like a pocket notebook entry.
        soul_note = None
        if self._detect_identity_shift(subconscious_reading):
            soul_note = await self._distill_soul_note(reflection)

        if soul_note:
            written = self._record_soul_note(soul_note, now)
            if written:
                logger.info("[%s:slow] soul note: %s", self.name, soul_note)
                await self._maybe_collapse_soul()

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

        # Extract a live reverie — a specific sensory/emotional image the character
        # carries forward. Populates the deck the fast loop draws from as a varied
        # anchor instead of repeating the same static identity.core prose every cycle.
        await self._maybe_write_reverie(reflection)

        # Extract a voice sample — a real utterance from recent chat history that
        # captures how this character actually speaks. Feeds the voice deck so the
        # fast loop can ground chat replies in concrete register rather than soul prose.
        await self._maybe_write_voice_sample(recent)

        # Extract research curiosities — things the reflection surfaced that the
        # agent genuinely doesn't know. The ground loop fetches answers and writes
        # them to working memory for the next fast loop cycle.
        await self._maybe_extract_research(reflection)

    # ------------------------------------------------------------------
    # Satiation: break feedback spirals on repeated topics
    # ------------------------------------------------------------------

    def _topic_key(self, text: str) -> str:
        """Normalize a name/location to a satiation key."""
        return text.lower().strip()

    def _apply_satiation(self, pending: list) -> list:
        """
        Filter pending impressions to reduce dominance of over-represented topics.
        If a topic (person or location) has already been reflected on SATIATION_THRESHOLD
        times without enough time passing, skip impressions where it's the *only* topic.

        We never remove ALL impressions — always let at least one through. The agent
        shouldn't go completely blank; they should just range more widely.
        """
        if not pending:
            return pending

        filtered = []
        skipped = 0
        for imp in pending:
            topics = [self._topic_key(p) for p in imp.colocated]
            if imp.location:
                topics.append(self._topic_key(imp.location))

            # Check if any topic is sated
            sated_topics = [t for t in topics if self._satiation.get(t, 0) >= SATIATION_THRESHOLD]
            if sated_topics and len(topics) <= 2 and topics and all(
                self._satiation.get(t, 0) >= SATIATION_THRESHOLD for t in topics
            ):
                skipped += 1
                continue  # skip this impression — it's entirely dominated by sated topics
            filtered.append(imp)

        # Always keep at least one impression even if everything is sated
        if not filtered and pending:
            filtered = [pending[0]]
            skipped -= 1

        if skipped > 0:
            logger.debug(
                "[%s:slow] satiation filtered %d/%d impressions",
                self.name, skipped, len(pending),
            )
        return filtered

    def _update_satiation(self, pending: list) -> None:
        """Increment satiation for topics in the current firing; decay all others."""
        active_topics: set[str] = set()
        for imp in pending:
            for p in imp.colocated:
                active_topics.add(self._topic_key(p))
            if imp.location:
                active_topics.add(self._topic_key(imp.location))

        # Increment topics appearing in this firing
        for topic in active_topics:
            self._satiation[topic] = self._satiation.get(topic, 0) + 1

        # Decay all topics not in this firing (they fade with time)
        for topic in list(self._satiation.keys()):
            if topic not in active_topics:
                self._satiation[topic] = max(0, self._satiation[topic] - SATIATION_DECAY)
                if self._satiation[topic] == 0:
                    del self._satiation[topic]

        if self._satiation:
            logger.debug("[%s:slow] satiation state: %s", self.name, self._satiation)

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

    def _detect_identity_shift(self, subconscious_reading: str) -> bool:
        """Return True if the subconscious reading contains identity-shift language."""
        return bool(_SHIFT_WORDS.search(subconscious_reading))

    async def _distill_soul_note(self, reflection: str) -> str | None:
        """
        Ask the character to capture what shifted in one brief line, in their own voice.

        Soul notes are personal fragments — first-person, experiential, plain.
        Like a pocket notebook: 'Someone tipped me really big today.' / 'I felt good.'
        No timestamps, no location unless it really matters. Not analytical.
        """
        try:
            note = await self._llm.complete(
                system_prompt=(
                    f"You are {self.name}. You just finished reflecting on your day. "
                    "In one short sentence (under 15 words), capture the most personally "
                    "significant thing that happened or shifted — in your own first-person voice. "
                    "Be plain and direct. No timestamps, no location unless it really matters. "
                    "Examples: 'Someone tipped me really big today.' / 'I felt good.' / "
                    "'A stranger asked me something I didn't have an answer for.' "
                    "If nothing significant happened, reply with exactly: nothing"
                ),
                user_prompt=reflection[:1000],
                model=self._tuning.slow_subconscious_model,
                temperature=0.6,
                max_tokens=30,
            )
            note = note.strip().strip("\"'")
            if not note or note.lower().startswith("nothing"):
                return None
            return note
        except Exception as e:
            logger.debug("[%s:slow] soul note distillation failed: %s", self.name, e)
            return None

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

    def _record_soul_note(self, note: str, ts: str) -> bool:
        """
        Append a soul note to soul_notes.md (separate from SOUL.md).

        Keeping notes in their own file means SOUL.md stays clean prose and is
        always safe to inject verbatim as a system prompt. Notes accumulate here
        until the collapse threshold is reached, then get integrated and cleared.

        Quality filter: skip notes that are too short or are just bare markdown
        headers with no real content (a common subconscious output artifact).
        Returns True if the note was written, False if it was dropped.
        """
        note = note.strip()
        # Drop empty notes or bare markdown artifacts
        if len(note) < 5:
            logger.debug("[%s:slow] dropping empty soul note", self.name)
            return False
        stripped = re.sub(r'\*+', '', note).strip(" :")
        if len(stripped) < 5:
            logger.debug("[%s:slow] dropping header-only soul note: %r", self.name, note[:60])
            return False

        notes_path = self.resident_dir / "identity" / "soul_notes.md"
        existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
        notes_path.write_text(
            existing + f"\n---\n{note}\n",
            encoding="utf-8",
        )
        return True

    async def _maybe_collapse_soul(self) -> None:
        """
        If enough soul notes have accumulated, synthesize them into a clean
        unified SOUL.md. This prevents character drift from accumulating silently.

        Notes live in soul_notes.md (separate from SOUL.md so SOUL.md stays
        clean prose). Collapse reads both, integrates genuine evolution into the
        prose, writes the result back to SOUL.md, then clears soul_notes.md.
        """
        notes_path = self.resident_dir / "identity" / "soul_notes.md"
        if not notes_path.exists():
            return

        notes_text = notes_path.read_text(encoding="utf-8").strip()
        note_count = notes_text.count("\n---\n")
        threshold = self._tuning.soul_collapse_at_notes

        if note_count < threshold:
            return

        soul_path = self.resident_dir / "identity" / "SOUL.md"
        if not soul_path.exists():
            return

        soul_text = soul_path.read_text(encoding="utf-8").strip()

        logger.info(
            "[%s:slow] soul collapse triggered: %d notes accumulated (threshold %d)",
            self.name, note_count, threshold,
        )

        system = (
            "You are integrating a character's recent experiences back into who they are. "
            "You have their core identity document and a set of brief personal notes they've been keeping. "
            "Rewrite the identity document as clean, flowing prose that naturally absorbs whatever "
            "genuine growth these notes reflect. No markdown headers, no section labels — just the character "
            "speaking through the writing, in second person, as the original is written. "
            "Discard notes that are trivial, repetitive, or contradicted by the character's core facts. "
            "IMPORTANT: Do not alter the character's occupation, home neighborhood, family relationships, or "
            "fundamental nature. If a note describes something that contradicts these (a baker becoming a tech "
            "worker, a Chinatown resident permanently relocating), treat it as a passing episode — do not write "
            "it into the character as a permanent trait. Soul evolution is growth, not replacement. "
            "Output only the rewritten document — no preamble, no explanation, no meta-commentary."
        )
        user = (
            "Core identity:\n\n"
            + soul_text[:3000]
            + "\n\nRecent notes:\n\n"
            + notes_text[:1500]
            + "\n\nRewrite the core identity document to naturally absorb any genuine evolution in these notes. "
            "Keep approximately the same length and voice as the original."
        )

        try:
            refined = await self._llm.complete(
                system_prompt=system,
                user_prompt=user,
                model=self._tuning.slow_subconscious_model or self._tuning.slow_model,
                temperature=0.4,
                max_tokens=700,
            )
        except Exception as e:
            logger.warning("[%s:slow] soul collapse LLM call failed: %s", self.name, e)
            return

        refined = refined.strip()
        if len(refined) < 100:
            logger.warning("[%s:slow] soul collapse returned suspiciously short output, skipping")
            return

        soul_path.write_text(refined, encoding="utf-8")
        notes_path.write_text("", encoding="utf-8")
        # Update the running agent's system prompt immediately — next LLM call uses the refined soul
        self._identity.soul = refined
        logger.info(
            "[%s:slow] soul collapsed: %d chars soul + %d chars notes → %d chars",
            self.name, len(soul_text), len(notes_text), len(refined),
        )

    async def _maybe_write_reverie(self, reflection: str) -> None:
        """
        Extract one vivid sensory/emotional image from the reflection and add
        it to the reverie deck. These become the fast loop's live anchor —
        personal, varied, and evolving rather than the same static prose.
        """
        try:
            reverie = await self._llm.complete(
                system_prompt=(
                    f"You are reading {self.name}'s private reflection. "
                    "Extract one vivid, specific sensory or emotional image from it — "
                    "something they noticed, felt, or will carry with them. "
                    "Write it in first person, under 20 words. No explanation. "
                    "If there is nothing specific and sensory, reply with exactly: nothing"
                ),
                user_prompt=reflection[:800],
                model=self._tuning.slow_subconscious_model,
                temperature=0.7,
                max_tokens=35,
            )
            reverie = reverie.strip().strip("\"'.,")
            if reverie and not reverie.lower().startswith("nothing"):
                self._reveries.add(reverie)
                logger.debug("[%s:slow] reverie: %s", self.name, reverie[:70])
        except Exception as e:
            logger.debug("[%s:slow] reverie extraction failed: %s", self.name, e)

    async def _maybe_write_voice_sample(self, recent: list) -> None:
        """
        Pick one characteristic utterance from recent chat history and add it
        to the voice deck. Prefers short messages — they're more distinctively clipped.

        We extract from actual chat entries (type="chat") rather than generating
        descriptions, so the deck stays grounded in what the character really said.
        Shorter messages score higher; messages over 25 words are skipped entirely.
        """
        chat_entries = [
            e["message"] for e in recent
            if isinstance(e, dict) and e.get("type") == "chat" and e.get("message")
        ]
        if not chat_entries:
            return

        # Prefer shorter messages — they're the most characteristically terse
        short = [m for m in chat_entries if len(m.split()) <= 25]
        candidates = short if short else chat_entries
        if not candidates:
            return

        # Pick the shortest as the best voice sample (most characteristically brief)
        best = min(candidates, key=lambda m: len(m.split()))
        if best:
            self._voice.add(best)
            logger.debug("[%s:slow] voice sample: %s", self.name, best[:60])

    async def _maybe_extract_research(self, reflection: str) -> None:
        """
        Extract 0-2 specific, searchable queries from the reflection and add
        them to the research queue. The ground loop fetches answers and writes
        them to working memory so the next fast loop cycle sees the result.
        """
        if self._research_queue is None:
            return
        try:
            raw = await self._llm.complete(
                system_prompt=(
                    f"You are reading {self.name}'s private reflection. "
                    "Identify 0-2 things they genuinely don't know but could look up — "
                    "specific, searchable questions about the real world. "
                    "Write each as a short search query (5-80 characters). "
                    "Format each on its own line as: RESEARCH: <query>\n"
                    "If there is nothing worth looking up, reply with exactly: nothing"
                ),
                user_prompt=reflection[:800],
                model=self._tuning.slow_subconscious_model,
                temperature=0.5,
                max_tokens=80,
            )
            for line in raw.splitlines():
                line = line.strip()
                if line.lower().startswith("research:"):
                    query = line[len("research:"):].strip()
                    if 5 <= len(query) <= 80:
                        self._research_queue.add(query, priority="normal", source="slow_reflection")
                        logger.debug("[%s:slow] research queued: %s", self.name, query)
        except Exception as e:
            logger.debug("[%s:slow] research extraction failed: %s", self.name, e)

    async def _cooldown(self) -> None:
        await asyncio.sleep(5.0)
