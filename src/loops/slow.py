from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.identity.loader import IdentityLoader, ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.memory.provisional import ProvisionalScratchpad
from src.memory.retrieval import LongTermMemory
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient, scene_to_prose, world_facts_to_prose

logger = logging.getLogger(__name__)

# Bracketed tag patterns the slow loop may naturally produce
_RE_ACTION = re.compile(r'\[ACTION:\s*(.+?)\]', re.IGNORECASE | re.DOTALL)
_RE_LETTER = re.compile(r'\[LETTER TO:\s*(\w+)\s*\|\s*(.+?)\]', re.IGNORECASE | re.DOTALL)
_RE_SOUL   = re.compile(r'\[SOUL:\s*(.+?)\]', re.IGNORECASE | re.DOTALL)


class SlowLoop(BaseLoop):
    """
    Reflective processing loop. Fires when enough impressions accumulate,
    or as a fallback timer.

    The agent receives their full self (SOUL.md), what's been happening
    recently, what they've been noticing, and what the world remembers.
    They write what they think and do.

    Lightweight bracketed tags at the end of their response let the
    framework act on their decisions — but the tags are optional and
    feel like marginalia, not API calls.
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
        self._ww = ww_client
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
                return  # enough has happened — reflect

        # Fallback timer expired — reflect anyway
        logger.debug("[%s:slow] fallback timer fired", self.name)

    # ------------------------------------------------------------------
    # Context: full self + recent history + impressions + world facts
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict:
        scene = await self._ww.get_scene(self._session_id)
        pending = self._provisional.pending_impressions()
        recent = self._working.all()

        # Build a query for world fact retrieval from current context
        query_tags = [scene.location] + [p.name for p in scene.present]
        query_text = f"{scene.location} " + " ".join(p.name for p in scene.present)

        world_facts = []
        try:
            world_facts = await self._ww.get_world_facts(query_text, self._session_id, limit=5)
        except Exception as e:
            logger.debug("[%s:slow] world facts unavailable: %s", self.name, e)

        long_term = self._long_term.retrieve(query_tags, limit=5)

        return {
            "scene": scene,
            "pending": pending,
            "recent": recent,
            "world_facts": world_facts,
            "long_term": long_term,
        }

    async def _should_act(self, context: dict) -> bool:
        return True  # slow loop always reflects when triggered

    # ------------------------------------------------------------------
    # Decide and execute
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        scene = context["scene"]
        pending = context["pending"]
        recent = context["recent"]
        world_facts = context["world_facts"]
        long_term = context["long_term"]

        # Build the user prompt — naturalized, no system vocabulary
        prompt_parts: list[str] = []

        # Where they are now
        prompt_parts.append(scene_to_prose(scene, self.name))

        # What's been happening lately (working memory as prose)
        if recent:
            event_lines = []
            for e in recent[-self._tuning.slow_max_context_events:]:
                if isinstance(e, dict) and e.get("action"):
                    event_lines.append(e["action"])
                elif isinstance(e, dict) and e.get("summary"):
                    event_lines.append(e["summary"])
            if event_lines:
                prompt_parts.append("Recently: " + " ".join(event_lines[-5:]))

        # What they've been noticing (provisional impressions as prose)
        impressions_prose = self._provisional.pending_as_prose()
        if impressions_prose:
            prompt_parts.append(impressions_prose)

        # What the world remembers (world facts, surfaced naturally)
        if world_facts:
            facts_prose = world_facts_to_prose(world_facts)
            if facts_prose:
                prompt_parts.append(facts_prose)

        # Personal long-term memories (retrieved by tag)
        if long_term:
            memory_lines = [m.content for m in long_term if m.content]
            if memory_lines:
                prompt_parts.append("\n".join(memory_lines))

        # Minimal guidance on optional output structure — framed as marginalia
        prompt_parts.append(
            "\nIf you act in the world: [ACTION: what you do]\n"
            "If you want to write to someone: [LETTER TO: name | your opening]\n"
            "If something has shifted in who you are: [SOUL: one sentence]"
        )

        user_prompt = "\n\n".join(prompt_parts)

        response = await self._llm.complete(
            system_prompt=self._identity.soul,
            user_prompt=user_prompt,
            model=self._tuning.slow_model,
            temperature=self._tuning.slow_temperature,
            max_tokens=self._tuning.slow_max_tokens,
        )

        await self._process_response(response, scene, pending)

    async def _process_response(self, response: str, scene, pending) -> None:
        now = datetime.now(timezone.utc).isoformat()

        # Extract structured tags (optional — present only if agent used them)
        action_match = _RE_ACTION.search(response)
        letter_match = _RE_LETTER.search(response)
        soul_match   = _RE_SOUL.search(response)

        # The reflection itself (strip tags for the log)
        reflection = re.sub(r'\[(?:ACTION|LETTER TO|SOUL):[^\]]+\]', '', response, flags=re.IGNORECASE).strip()

        # 1. Take world action if specified
        if action_match:
            action_text = action_match.group(1).strip()
            try:
                result = await self._ww.post_action(self._session_id, action_text)
                logger.info("[%s:slow] acted: %s", self.name, action_text[:80])
                self._working.append({
                    "ts": now, "loop": "slow", "location": scene.location,
                    "action": action_text,
                    "narrative": result.narrative[:200] if result.narrative else "",
                })
            except Exception as e:
                logger.warning("[%s:slow] action failed: %s", self.name, e)

        # 2. Stage letter draft if specified
        if letter_match:
            recipient = letter_match.group(1).strip()
            body = letter_match.group(2).strip()
            self._stage_letter_draft(recipient, body)
            logger.info("[%s:slow] staged letter to %s", self.name, recipient)

        # 3. Update SOUL.md if the agent signaled a shift in identity
        if soul_match:
            soul_note = soul_match.group(1).strip()
            self._record_soul_note(soul_note, now)
            logger.info("[%s:slow] soul note: %s", self.name, soul_note)

        # 4. Save decision log entry
        self._decision_count += 1
        decision_path = self._decisions_dir / f"decision_{self._decision_count}.json"
        decision_path.write_text(json.dumps({
            "ts": now,
            "loop": "slow",
            "location": scene.location,
            "reflection": reflection,
            "action": action_match.group(1).strip() if action_match else None,
            "letter_to": letter_match.group(1).strip() if letter_match else None,
            "soul_note": soul_match.group(1).strip() if soul_match else None,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # 5. Process provisional impressions — they were the context, now integrate them
        for imp in pending:
            # Archive with the reflection as interpretation
            self._provisional.archive(imp, reflection[:200])

        # 6. Store a long-term memory from this reflection if substantive
        if len(reflection) > 50:
            tags = [scene.location] + [p.name for p in scene.present]
            self._long_term.store(reflection[:400], tags=tags, source="slow_reflection")

    def _stage_letter_draft(self, recipient: str, body: str) -> None:
        drafts_dir = self.resident_dir / "letters" / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        draft_path = drafts_dir / f"draft_{ts}_{recipient}.md"
        draft_path.write_text(
            f"To: {recipient}\nStaged-At: {ts}\n\n{body}",
            encoding="utf-8"
        )

    def _record_soul_note(self, note: str, ts: str) -> None:
        """
        Append a soul note to SOUL.md. The slow loop can propose edits
        to its own identity — small, earned, rare.

        We append a dated note rather than rewriting SOUL.md wholesale,
        which is safer than letting a single reflection overwrite the whole document.
        The operator can periodically review and integrate notes into the main text.
        """
        soul_path = self.resident_dir / "identity" / "SOUL.md"
        if soul_path.exists():
            current = soul_path.read_text(encoding="utf-8")
            soul_path.write_text(
                current + f"\n\n---\n*{ts[:10]}:* {note}",
                encoding="utf-8"
            )

    async def _cooldown(self) -> None:
        # Slow loop cooldown is minimal — the trigger mechanism provides the pacing
        await asyncio.sleep(5.0)
