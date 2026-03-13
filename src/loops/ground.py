"""
ground.py — Real-world grounding loop.

Fires every ~35 minutes (configurable). Fetches current SF time and weather
from the worldweaver backend, then generates a brief naturalistic observation
that the agent "experiences" — glancing at a phone, noticing the fog, feeling
the afternoon heat. The observation lands in working memory as a grounding
impression so every other loop naturally incorporates it without needing to
know anything about the grounding data structure.

The LLM never sees raw API fields. It sees: who the agent is, where they are,
and a one-line real-world fact. It produces prose. That's it.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from src.identity.loader import ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient

logger = logging.getLogger(__name__)


class GroundLoop(BaseLoop):
    """
    Ambient awareness loop. Injects real SF time and weather into working
    memory as naturalistic prose impressions.

    The agent doesn't receive a structured payload — it experiences a moment:
    "Rosa glances at her phone. 9:47 AM, Thursday. The fog is sitting heavy
    on Valencia this morning."
    """

    def __init__(
        self,
        identity: ResidentIdentity,
        resident_dir: Path,
        ww_client: WorldWeaverClient,
        llm: InferenceClient,
        session_id: str,
        working_memory: WorkingMemory,
    ):
        super().__init__(identity.name, resident_dir)
        self._identity = identity
        self._ww = ww_client
        self._llm = llm
        self._session_id = session_id
        self._working = working_memory
        self._tuning = identity.tuning

    # ------------------------------------------------------------------
    # Trigger: real-time interval (~35 minutes with ±15% jitter)
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        minutes = self._tuning.ground_minutes
        jitter = random.uniform(-minutes * 0.15, minutes * 0.15)
        await asyncio.sleep((minutes + jitter) * 60)

    # ------------------------------------------------------------------
    # Context: grounding data + current location
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict:
        grounding: dict = {}
        location = "somewhere in the city"
        news: list[str] = []

        try:
            grounding = await self._ww.get_grounding()
        except Exception as e:
            logger.warning("[%s:ground] grounding fetch failed: %s", self.name, e)

        try:
            scene = await self._ww.get_scene(self._session_id)
            location = scene.location
        except Exception as e:
            logger.warning("[%s:ground] scene fetch failed: %s", self.name, e)

        try:
            news = await self._ww.get_news()
        except Exception as e:
            logger.debug("[%s:ground] news fetch failed: %s", self.name, e)

        return {"grounding": grounding, "location": location, "news": news}

    async def _should_act(self, context: dict) -> bool:
        return bool(context.get("grounding"))

    # ------------------------------------------------------------------
    # Generate grounding moment
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        grounding = context["grounding"]
        location = context["location"]
        news = context.get("news", [])
        name = self._identity.name

        datetime_str = grounding.get("datetime_str", "")
        weather_desc = grounding.get("weather_description") or grounding.get(
            "weather", ""
        )

        world_line = datetime_str
        if weather_desc:
            world_line += f". Weather: {weather_desc}"

        # Include at most one headline — enough to make the world feel alive
        # without overwhelming a sensory grounding moment.
        news_line = ""
        if news:
            news_line = f"\n\nIn the news today: {news[0]}."

        user_prompt = (
            f"You are {name}, currently at {location}.\n\n"
            f"Right now in San Francisco: {world_line}.{news_line}\n\n"
            f"In one or two sentences, describe what {name} briefly notices about "
            f"the world at this moment — a glance at a phone, a look out the window, "
            f"a feeling in the air, the quality of light. "
            f"Be specific and sensory. No drama. Just what's there."
        )

        try:
            response = await self._llm.complete(
                system_prompt=self._identity.soul_with_context,
                user_prompt=user_prompt,
                model=self._tuning.fast_model,
                temperature=self._tuning.ground_temperature,
                max_tokens=80,
            )
        except Exception as e:
            logger.warning("[%s:ground] LLM call failed: %s", self.name, e)
            return

        observation = response.strip()
        if not observation:
            return

        logger.info("[%s:ground] %s", self.name, observation[:100])

        self._working.append(
            {
                "type": "grounding",
                "text": observation,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _cooldown(self) -> None:
        pass  # interval handled entirely in _wait_for_trigger
