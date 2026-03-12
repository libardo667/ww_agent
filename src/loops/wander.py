"""
wander.py — Spatial-consciousness loop with GPS-style persistent routing.

Two-phase operation:

NAVIGATION phase (active route in memory/active_route.json):
    Each wander tick advances one hop toward the saved destination via post_map_move.
    The server does BFS — we just keep sending the final destination.
    When route_remaining is empty we've arrived; clear the route file.

PLANNING phase (no active route):
    Build a rich introspection prompt: grounding (time/weather), recent events,
    available locations, diegetic framing ("check your phone / transit directory /
    ask someone"). LLM returns ONLY a destination name or "stay". If a destination
    is chosen, immediately advance the first hop and save the route.

The LLM is never involved during navigation — only during planning. Movement is
always via the explicit map API, never parsed from narrative text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from src.identity.loader import ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient

logger = logging.getLogger(__name__)

_ROUTE_FILENAME = "active_route.json"


def _normalize(name: str) -> str:
    """Convert a display name to the location key suffix used in the graph."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _adjacent_names(location_graph: dict, current_location: str) -> list[str]:
    """Return display names of locations directly connected to current_location.

    Handles both location: and landmark: node key prefixes.
    """
    nodes = location_graph.get("nodes", [])
    edges = location_graph.get("edges", [])
    name_to_key = {n["name"]: n["key"] for n in nodes if n.get("key") and n.get("name")}
    current_key = name_to_key.get(current_location) or f"location:{_normalize(current_location)}"
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
    Spatial-consciousness loop with GPS-style persistent routing.

    Planning phase: LLM picks a destination.
    Navigation phase: hop-by-hop toward that destination, no LLM involved.
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
        self._route_path = resident_dir / "memory" / _ROUTE_FILENAME

    # ------------------------------------------------------------------
    # Route file helpers
    # ------------------------------------------------------------------

    def _load_route(self) -> Optional[dict]:
        """Return {destination, remaining} if an active route exists."""
        if not self._route_path.exists():
            return None
        try:
            data = json.loads(self._route_path.read_text(encoding="utf-8"))
            if data.get("destination"):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _save_route(self, destination: str, remaining: list[str]) -> None:
        self._route_path.parent.mkdir(parents=True, exist_ok=True)
        self._route_path.write_text(
            json.dumps({"destination": destination, "remaining": remaining}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _clear_route(self) -> None:
        if self._route_path.exists():
            try:
                self._route_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Trigger: pure dwell timer
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        wander_seconds = self._tuning.wander_seconds
        jitter = random.uniform(0.0, min(30.0, wander_seconds * 0.1))
        await asyncio.sleep(wander_seconds + jitter)

    # ------------------------------------------------------------------
    # Context: scene + grounding + recent events + active route
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict:
        scene = await self._ww.get_scene(self._session_id)
        graph = scene.location_graph
        nodes = graph.get("nodes", [])

        adjacent = _adjacent_names(graph, scene.location)

        all_names = [
            n["name"] for n in nodes if n.get("name") and n["name"] != scene.location
        ]
        far_names = [n for n in all_names if n not in adjacent]
        far_sample = (
            random.sample(far_names, min(4, len(far_names))) if far_names else []
        )

        # Grounding: most recent entry
        grounding_text = ""
        for entry in reversed(self._working.recent(8)):
            if entry.get("type") == "grounding":
                grounding_text = entry.get("text", "")
                break

        # Recent events: last 3 non-grounding entries as prose
        recent_events: list[str] = []
        for entry in reversed(self._working.recent(6)):
            if entry.get("type") != "grounding" and entry.get("text"):
                recent_events.append(entry["text"])
            if len(recent_events) >= 3:
                break
        recent_events.reverse()

        return {
            "current_location": scene.location,
            "adjacent": adjacent,
            "far_options": far_sample,
            "all_options": adjacent + far_sample,
            "grounding_text": grounding_text,
            "recent_events": recent_events,
            "active_route": self._load_route(),
        }

    async def _should_act(self, context: dict) -> bool:
        return True

    # ------------------------------------------------------------------
    # Decide: navigate (follow route) or plan (pick destination)
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        active_route = context["active_route"]

        if active_route:
            await self._navigate(active_route, context["current_location"])
        else:
            await self._plan_and_depart(context)

    # ------------------------------------------------------------------
    # Navigation phase: advance one hop, update/clear route
    # ------------------------------------------------------------------

    async def _navigate(self, route: dict, current_location: str) -> None:
        destination = route["destination"]

        if current_location.lower() == destination.lower():
            logger.info("[%s:wander] arrived at destination %s", self.name, destination)
            self._clear_route()
            return

        logger.info("[%s:wander] en route to %s — advancing one hop", self.name, destination)
        try:
            result = await self._ww.post_map_move(self._session_id, destination)
        except Exception as e:
            logger.warning("[%s:wander] hop failed: %s", self.name, e)
            return

        if result.get("moved"):
            arrived_at = result.get("to_location", destination)
            remaining = result.get("route_remaining", [])
            logger.info("[%s:wander] hopped to %s", self.name, arrived_at)
            if not remaining or arrived_at.lower() == destination.lower():
                logger.info("[%s:wander] route complete — arrived at %s", self.name, destination)
                self._clear_route()
            else:
                self._save_route(destination, remaining)
        else:
            logger.debug("[%s:wander] move returned moved=false: %s", self.name, result)
            # Route may have become invalid (snapped, graph changed). Clear and replan next cycle.
            self._clear_route()

    # ------------------------------------------------------------------
    # Planning phase: LLM picks destination, first hop immediately
    # ------------------------------------------------------------------

    async def _plan_and_depart(self, context: dict) -> None:
        current = context["current_location"]
        adjacent = context["adjacent"]
        far_options = context["far_options"]
        all_options = context["all_options"]
        grounding_text = context["grounding_text"]
        recent_events = context["recent_events"]
        name = self._identity.name

        # --- Build prompt ---
        nearby_str = (
            "Directly reachable: " + ", ".join(adjacent) + "."
            if adjacent
            else "No adjacent locations mapped yet."
        )
        far_str = ("Further away: " + ", ".join(far_options) + ".") if far_options else ""

        grounding_line = f"Right now: {grounding_text}" if grounding_text else ""

        if recent_events:
            events_block = "Recent: " + " / ".join(recent_events)
        else:
            events_block = ""

        context_lines = [line for line in [grounding_line, events_block] if line]
        context_block = "\n".join(context_lines)

        diegetic_hint = (
            "To decide, they might check their phone for transit, glance at a neighborhood map, "
            "think about what errand is overdue, or simply follow a habit."
        )

        user_prompt = (
            f"{name} is at {current}.\n"
            + (f"{context_block}\n\n" if context_block else "\n")
            + f"{nearby_str}"
            + (f" {far_str}" if far_str else "")
            + f"\n\n{diegetic_hint}\n\n"
            f"Where does {name} feel like going next — or stay put?\n\n"
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
            logger.debug("[%s:wander] decided to stay at %s", self.name, current)
            return

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

        logger.info("[%s:wander] planning route to %s", self.name, destination)
        try:
            result = await self._ww.post_map_move(self._session_id, destination)
        except Exception as e:
            logger.warning("[%s:wander] first hop failed: %s", self.name, e)
            return

        if result.get("moved"):
            arrived_at = result.get("to_location", destination)
            remaining = result.get("route_remaining", [])
            logger.info("[%s:wander] first hop → %s", self.name, arrived_at)
            if not remaining or arrived_at.lower() == destination.lower():
                logger.info("[%s:wander] single-hop trip complete — at %s", self.name, destination)
                # No route to save — arrived immediately
            else:
                self._save_route(destination, remaining)
        else:
            logger.debug(
                "[%s:wander] first hop returned moved=false: %s", self.name, result
            )

    # ------------------------------------------------------------------
    # Cooldown: dwell timer IS the cooldown
    # ------------------------------------------------------------------

    async def _cooldown(self) -> None:
        pass
