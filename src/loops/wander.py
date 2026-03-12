"""
wander.py — Route keeper loop.

The fast loop now handles all movement planning and reactive movement via its
'move:' slug. The wander loop's sole job is to advance an existing route when
the agent is in navigation mode and hasn't moved recently.

No LLM is involved in the wander loop. It reads active_route.json (written by
the fast loop), calls post_map_move with the saved final destination (the server
does BFS for the next hop), and clears the route on arrival or failure.

This keeps long-distance journeys progressing even if the fast loop is occupied
with events at an intermediate location. The fast loop can also re-route at any
time by overwriting active_route.json with a new destination.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Optional

from src.identity.loader import ResidentIdentity
from src.loops.base import BaseLoop
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient

logger = logging.getLogger(__name__)

_ROUTE_FILENAME = "active_route.json"


class WanderLoop(BaseLoop):
    """
    Route keeper. Advances an active route one hop per wander tick.
    Fires only when an active_route.json exists — otherwise waits quietly.
    """

    def __init__(
        self,
        identity: ResidentIdentity,
        resident_dir: Path,
        ww_client: WorldWeaverClient,
        session_id: str,
        working_memory: WorkingMemory,
    ):
        super().__init__(identity.name, resident_dir)
        self._identity = identity
        self._ww = ww_client
        self._session_id = session_id
        self._working = working_memory
        self._tuning = identity.tuning
        self._route_path = resident_dir / "memory" / _ROUTE_FILENAME

    # ------------------------------------------------------------------
    # Trigger: dwell timer
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        wander_seconds = self._tuning.wander_seconds
        jitter = random.uniform(0.0, min(30.0, wander_seconds * 0.1))
        await asyncio.sleep(wander_seconds + jitter)

    # ------------------------------------------------------------------
    # Context: just the active route and current location
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict:
        route = self._load_route()
        current_location = ""
        if route:
            try:
                scene = await self._ww.get_scene(self._session_id)
                current_location = scene.location
            except Exception as e:
                logger.debug("[%s:wander] scene fetch failed: %s", self.name, e)
        return {"route": route, "current_location": current_location}

    async def _should_act(self, context: dict) -> bool:
        # Only fire if there is an active route to advance
        return context.get("route") is not None

    # ------------------------------------------------------------------
    # Advance one hop along the saved route
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        route = context["route"]
        current_location = context["current_location"]
        destination = route["destination"]

        if current_location and current_location.lower() == destination.lower():
            logger.info("[%s:wander] already at destination %s — clearing route", self.name, destination)
            self._clear_route()
            return

        logger.info("[%s:wander] advancing route toward %s", self.name, destination)
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
            # Route became invalid (location snapped, graph changed). Clear so fast loop replans.
            logger.debug("[%s:wander] move returned moved=false — clearing stale route", self.name)
            self._clear_route()

    async def _cooldown(self) -> None:
        pass  # wander_seconds IS the cooldown; handled in _wait_for_trigger

    # ------------------------------------------------------------------
    # Route file helpers
    # ------------------------------------------------------------------

    def _load_route(self) -> Optional[dict]:
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
