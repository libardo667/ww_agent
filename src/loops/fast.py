from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.identity.loader import ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.memory.provisional import ProvisionalScratchpad
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient, scene_to_prose

logger = logging.getLogger(__name__)


class FastLoop(BaseLoop):
    """
    Scene-local reaction loop. Fires when something happens nearby.

    The agent receives only: who they are (SOUL.md) and what's in front of
    them right now (scene as prose). They write what they do. That's it.

    No memory beyond the last few events. No letters. No reflection.
    Just presence.
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
    ):
        super().__init__(identity.name, resident_dir)
        self._identity = identity
        self._ww = ww_client
        self._llm = llm
        self._session_id = session_id
        self._working = working_memory
        self._provisional = provisional
        self._tuning = identity.tuning
        self._last_event_ts: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Trigger: poll for new events at our location
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        poll_interval = min(self._tuning.fast_cooldown_seconds, 30.0)

        while True:
            await asyncio.sleep(poll_interval)
            try:
                events = await self._ww.get_new_events(self._session_id, since=self._last_event_ts)
                if events:
                    self._last_event_ts = events[-1].ts
                    return  # something happened — fire
            except Exception as e:
                logger.debug("[%s:fast] event poll failed: %s", self.name, e)

    # ------------------------------------------------------------------
    # Context: just the scene, just now
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict:
        scene = await self._ww.get_scene(self._session_id)
        return {"scene": scene}

    async def _should_act(self, context: dict) -> bool:
        # Could check act_threshold against scene busyness, but for now: always act
        return True

    # ------------------------------------------------------------------
    # Decide and execute — the heart of the loop
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        scene = context["scene"]
        scene_prose = scene_to_prose(scene, self.name)

        # The system prompt is the character. Nothing else.
        system_prompt = self._identity.soul

        # The user prompt is the world as they experience it.
        # No instructions, no format requirements, no API vocabulary.
        # The agent writes what they do, and we submit that text.
        user_prompt = scene_prose

        response = await self._llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self._tuning.fast_model,
            temperature=self._tuning.fast_temperature,
            max_tokens=self._tuning.fast_max_tokens,
        )

        action = response.strip()

        if not action:
            logger.debug("[%s:fast] empty response, skipping", self.name)
            return

        # Check if agent flagged something as a notable impression
        # Format the agent might naturally use: text ending with (*)
        # We don't tell them this — it emerges if it does, and we parse it if it does.
        # Otherwise we extract a raw reaction from the action itself.
        impression_text, action_text = self._extract_impression(action)

        # Submit the action to the world
        try:
            result = await self._ww.post_action(self._session_id, action_text)
            logger.info("[%s:fast] acted: %s", self.name, action_text[:80])

            # Record in working memory
            self._working.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "loop": "fast",
                "location": scene.location,
                "action": action_text,
                "narrative": result.narrative[:200] if result.narrative else "",
            })

            # Write a provisional impression if the response surfaced something notable
            if impression_text or (result.narrative and self._seems_notable(result.narrative)):
                self._provisional.write_impression(
                    trigger=action_text,
                    raw_reaction=impression_text or result.narrative[:200],
                    location=scene.location,
                    colocated=[p.name for p in scene.present if p.name.lower() != self.name.lower()],
                )

        except Exception as e:
            logger.warning("[%s:fast] action failed: %s", self.name, e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_impression(self, text: str) -> tuple[str, str]:
        """
        If the agent marked something as notable with (*) at the end,
        extract it as the impression and return (impression, action).
        Otherwise return ("", text).

        We don't instruct the agent to use this format — it's a soft parse
        for if they naturally do something like it.
        """
        match = re.search(r'\(\*([^)]+)\)\s*$', text)
        if match:
            impression = match.group(1).strip()
            action = text[:match.start()].strip()
            return impression, action
        return "", text

    def _seems_notable(self, narrative: str) -> bool:
        """Rough heuristic: does the narrative suggest something worth flagging?"""
        notable_words = ["strange", "unexpected", "surprised", "odd", "familiar",
                         "uneasy", "wrong", "different", "changed", "recognized"]
        lower = narrative.lower()
        return any(w in lower for w in notable_words)

    async def _cooldown(self) -> None:
        await asyncio.sleep(self._tuning.fast_cooldown_seconds)
