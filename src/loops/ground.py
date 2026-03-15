"""
ground.py — Real-world grounding loop.

Fires every ~35 minutes (configurable). Fetches current SF time and weather
from the worldweaver backend, then generates a brief naturalistic observation
that the agent "experiences" — glancing at a phone, noticing the fog, feeling
the afternoon heat. The observation lands in working memory as a grounding
impression so every other loop naturally incorporates it without needing to
know anything about the grounding data structure.

Also consumes one item from the agent's research queue per cycle: fetches a
compact result packet from the web, distills it to prose, and writes it to
working memory as type="research" so the next fast loop cycle can react to it.

The LLM never sees raw API fields. It sees: who the agent is, where they are,
and a one-line real-world fact. It produces prose. That's it.
"""

from __future__ import annotations

import asyncio
import logging
import random
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.identity.loader import ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.memory.research_queue import ResearchQueue
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

    If the agent has pending research queries, one is consumed per cycle:
    fetched from the web, distilled, and written to working memory.
    """

    def __init__(
        self,
        identity: ResidentIdentity,
        resident_dir: Path,
        ww_client: WorldWeaverClient,
        llm: InferenceClient,
        session_id: str,
        working_memory: WorkingMemory,
        research_queue: ResearchQueue | None = None,
    ):
        super().__init__(identity.name, resident_dir)
        self._identity = identity
        self._ww = ww_client
        self._llm = llm
        self._session_id = session_id
        self._working = working_memory
        self._tuning = identity.tuning
        self._research_queue = research_queue

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
    # Generate grounding moment + consume one research item if queued
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

        # Consume one research item if the queue has anything pending
        if self._research_queue and len(self._research_queue) > 0:
            item = self._research_queue.pop_next()
            if item:
                await self._fetch_research(item)

    # ------------------------------------------------------------------
    # Research: pop one queued item, fetch, distil, write to working mem
    # ------------------------------------------------------------------

    async def _fetch_research(self, item: dict) -> None:
        """Fetch a web result for the queued query and write it to working memory."""
        query = item["query"]
        logger.info("[%s:ground] researching: %s", self.name, query)

        raw_text = await self._search_web(query)
        if not raw_text:
            logger.debug("[%s:ground] research: no result for %r", self.name, query)
            return

        try:
            distilled = await self._llm.complete(
                system_prompt=self._identity.soul_with_context,
                user_prompt=(
                    f"You just looked something up: {query}\n\n"
                    f"What you found:\n{raw_text[:1200]}\n\n"
                    f"In 1-3 sentences, note what's relevant or interesting to you as {self._identity.name}. "
                    f"Be specific. No editorializing."
                ),
                model=self._tuning.fast_model,
                temperature=0.4,
                max_tokens=100,
            )
        except Exception as e:
            logger.warning("[%s:ground] research distillation failed: %s", self.name, e)
            return

        distilled = distilled.strip()
        if distilled:
            self._working.append({
                "type": "research",
                "query": query,
                "result": distilled,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            logger.info("[%s:ground] research: %s", self.name, distilled[:120])

    async def _search_web(self, query: str) -> str:
        """
        Fetch a compact text result for query via DuckDuckGo Instant Answers.
        Falls back to top RelatedTopics snippets if AbstractText is empty.
        Returns empty string if nothing useful is found.
        """
        url = (
            "https://api.duckduckgo.com/?"
            + urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "t": "worldweaver",
            })
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, follow_redirects=True)
                data = resp.json()

            text = data.get("AbstractText") or ""
            if not text:
                topics = data.get("RelatedTopics", [])
                snippets = [
                    t["Text"] for t in topics[:4]
                    if isinstance(t, dict) and t.get("Text")
                ]
                text = " ".join(snippets)
            return text.strip()
        except Exception as e:
            logger.debug("[%s:ground] web search failed for %r: %s", self.name, query, e)
            return ""

    async def _cooldown(self) -> None:
        pass  # interval handled entirely in _wait_for_trigger
