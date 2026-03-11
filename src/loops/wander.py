"""
wander.py — Spatial-consciousness loop.

Fires on a dwell timer. The agent surfaces, looks at where it is, considers
adjacent locations plus a few far options, optionally notices recent grounding
context, and decides whether to stay or move.

The LLM responds with ONLY the destination name (or "stay"). No movement-verb
parsing needed — if the response matches a known location name, we call
post_map_move() directly. Clean, explicit, one-hop movement.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from pathlib import Path

from src.identity.loader import ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Convert a display name to the location key suffix used in the graph."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _adjacent_names(location_graph: dict, current_location: str) -> list[str]:
    """Return display names of locations directly connected to current_location."""
    nodes = location_graph.get("nodes", [])
    edges = location_graph.get("edges", [])
    current_key = f"location:{_normalize(current_location)}"
    adjacent_keys: set[str] = set()
    for e in edges:
        if e.get("from") == current_key:
            adjacent_keys.add(e["to"])
        elif e.get("to") == current_key:
            adjacent_keys.add(e["from"])
    key_to_name = {n["key"]: n["name"] for n in nodes if n.get("key") and n.get("name")}
    return [key_to_name[k] for k in adjacent_keys if k in key_to_name]


class WanderLoop(BaseLoop):
    """
    Spatial-consciousness loop. Fires on a dwell timer, not on events.

    Every `wander_seconds` the agent decides whether to stay or move. The LLM
    returns only a location name (or "stay"). Movement is executed via the
    explicit map API — no NL parsing needed.
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
    # Trigger: pure dwell timer
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        wander_seconds = self._tuning.wander_seconds
        jitter = random.uniform(0.0, min(30.0, wander_seconds * 0.1))
        await asyncio.sleep(wander_seconds + jitter)

    # ------------------------------------------------------------------
    # Context: where am I, what's nearby, what have I noticed lately?
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict:
        scene = await self._ww.get_scene(self._session_id)
        graph = scene.location_graph
        nodes = graph.get("nodes", [])

        adjacent = _adjacent_names(graph, scene.location)

        # Sample a few far locations for longer journeys
        all_names = [
            n["name"] for n in nodes if n.get("name") and n["name"] != scene.location
        ]
        far_names = [n for n in all_names if n not in adjacent]
        far_sample = (
            random.sample(far_names, min(4, len(far_names))) if far_names else []
        )

        # Most recent grounding impression (if any)
        grounding_text = ""
        for entry in reversed(self._working.recent(8)):
            if entry.get("type") == "grounding":
                grounding_text = entry.get("text", "")
                break

        return {
            "current_location": scene.location,
            "adjacent": adjacent,
            "far_options": far_sample,
            "grounding_text": grounding_text,
        }

    async def _should_act(self, context: dict) -> bool:
        return True

    # ------------------------------------------------------------------
    # Decide: stay or move?
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        current = context["current_location"]
        adjacent = context["adjacent"]
        far_options = context["far_options"]
        grounding_text = context["grounding_text"]

        all_options = adjacent + far_options
        name = self._identity.name

        if adjacent:
            nearby_str = "Directly nearby: " + ", ".join(adjacent) + "."
        else:
            nearby_str = "No adjacent locations mapped yet."

        if far_options:
            far_str = "Further away: " + ", ".join(far_options) + "."
        else:
            far_str = ""

        grounding_line = f"\nRight now: {grounding_text}" if grounding_text else ""

        user_prompt = (
            f"{name} is currently at {current}.{grounding_line}\n\n"
            f"{nearby_str}"
            + (f" {far_str}" if far_str else "")
            + f"\n\nWhere does {name} feel like going next, or should they stay?\n\n"
            f"Reply with ONLY the exact location name from the list above, or the single word: stay"
        )

        try:
            response = await self._llm.complete(
                system_prompt=self._identity.soul,
                user_prompt=user_prompt,
                model=self._tuning.fast_model,
                temperature=self._tuning.wander_temperature,
                max_tokens=20,
            )
        except Exception as e:
            logger.warning("[%s:wander] LLM call failed: %s", self.name, e)
            return

        decision = response.strip().strip(".,!\"'").strip()
        if not decision:
            return

        if decision.lower() == "stay":
            logger.debug("[%s:wander] staying at %s", self.name, current)
            return

        # Match against known options (case-insensitive)
        decision_lower = decision.lower()
        destination = next(
            (opt for opt in all_options if opt.lower() == decision_lower),
            None,
        )

        if not destination:
            logger.debug(
                "[%s:wander] unrecognized destination %r — staying", self.name, decision
            )
            return

        logger.info("[%s:wander] moving to %s", self.name, destination)
        try:
            result = await self._ww.post_map_move(self._session_id, destination)
            if result.get("moved"):
                logger.info(
                    "[%s:wander] arrived at %s",
                    self.name,
                    result.get("to_location", destination),
                )
            else:
                logger.debug(
                    "[%s:wander] move returned moved=false: %s", self.name, result
                )
        except Exception as e:
            logger.warning("[%s:wander] map move failed: %s", self.name, e)

    # ------------------------------------------------------------------
    # Cooldown: dwell timer IS the cooldown
    # ------------------------------------------------------------------

    async def _cooldown(self) -> None:
        pass
