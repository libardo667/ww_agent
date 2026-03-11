from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types — matches actual WorldWeaver API response shapes
# ---------------------------------------------------------------------------

@dataclass
class PresentCharacter:
    name: str
    role: str
    last_action: str
    last_seen: str  # ISO-8601


@dataclass
class RecentEvent:
    who: str
    summary: str
    ts: str  # ISO-8601


@dataclass
class SceneData:
    session_id: str
    location: str
    role: str
    present: list[PresentCharacter]
    recent_events_here: list[RecentEvent]
    location_graph: dict  # raw, used for navigation only — not surfaced to LLM


@dataclass
class TurnResult:
    narrative: str          # from /api/action: "narrative"; from /api/next: "text"
    choices: list[dict]
    vars: dict
    plausible: bool = True


@dataclass
class WorldFact:
    summary: str
    subject: str = ""
    predicate: str = ""
    value: str = ""
    confidence: float = 1.0


@dataclass
class Letter:
    filename: str
    body: str


@dataclass
class ChatMessage:
    id: int
    session_id: str
    display_name: str
    message: str
    ts: str  # ISO-8601


# ---------------------------------------------------------------------------
# Prose rendering — SceneData → natural language for LLM prompts
# The agent never sees raw API fields.
# ---------------------------------------------------------------------------

def scene_to_prose(scene: SceneData, character_name: str) -> str:
    """
    Transform structured scene data into natural prose for LLM context.
    No JSON field names, no API vocabulary, no raw arrays.

    Example output:
        You are in the Deeper Corridor. Casper is nearby — he set something
        down a few minutes ago and hasn't moved since. Elias left a while back.

        Recently here: A low hum started somewhere in the ceiling.
    """
    parts: list[str] = []

    # Location
    parts.append(f"You are in {scene.location}.")

    # Who's present (excluding self)
    others = [p for p in scene.present if p.name.lower() != character_name.lower()]
    if others:
        presence_parts = []
        for p in others:
            # Prefer role (character/player name) over name (session slug)
            display = p.role if p.role and p.role != p.name else p.name
            if p.last_action:
                presence_parts.append(f"{display} is here — {p.last_action.rstrip('.')}")
            else:
                presence_parts.append(f"{display} is here")
        parts.append(" ".join(presence_parts) + ".")
    else:
        parts.append("No one else is here right now.")

    # Recent events at this location
    if scene.recent_events_here:
        event_lines = [e.summary.rstrip(".") for e in scene.recent_events_here[:5]]
        parts.append("Recently: " + ". ".join(event_lines) + ".")

    return " ".join(parts)


def world_facts_to_prose(facts: list[WorldFact], limit: int = 5) -> str:
    """
    Render retrieved world facts as prose for slow loop context.
    Returns empty string if no facts. Facts are already summaries — just join them.
    """
    if not facts:
        return ""
    lines = [f.summary for f in facts[:limit] if f.summary]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class WorldClientError(Exception):
    pass


class WorldWeaverClient:
    """
    Async HTTP client for the WorldWeaver server.
    Shared across all residents. Stateless — all session context comes from caller.
    """

    def __init__(self, base_url: str, timeout_scene: float = 30.0, timeout_action: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._timeout_scene = timeout_scene
        self._timeout_action = timeout_action
        # Single shared connection pool
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Content-Type": "application/json"},
        )

    # ------------------------------------------------------------------
    # Health & World ID
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def wait_for_ready(self, timeout_seconds: float = 120.0, poll_interval: float = 2.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            if await self.health():
                return
            await asyncio.sleep(poll_interval)
        raise WorldClientError(f"WorldWeaver server not ready after {timeout_seconds}s")

    async def get_world_id(self) -> str | None:
        """Get the shared world ID. Returns None if not yet seeded."""
        resp = await self._get("/api/world/id", timeout=10.0)
        data = resp.json()
        return data.get("world_id") or None

    # ------------------------------------------------------------------
    # Session Bootstrap
    # ------------------------------------------------------------------

    async def bootstrap_session(
        self,
        session_id: str,
        world_id: str,
        world_theme: str,
        player_role: str,
        *,
        tone: str = "grounded, observational",
        description: str = "",
        entry_location: str = "",
    ) -> dict:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "world_id": world_id,
            "world_theme": world_theme,
            "player_role": player_role,
            "tone": tone,
            "bootstrap_source": "worldweaver-agent",
        }
        if description:
            payload["description"] = description
        if entry_location:
            payload["entry_location"] = entry_location

        resp = await self._post("/api/session/bootstrap", payload, timeout=60.0)
        return resp.json()

    # ------------------------------------------------------------------
    # Scene (fast + slow loops)
    # ------------------------------------------------------------------

    async def get_scene(self, session_id: str) -> SceneData:
        resp = await self._get_with_retry(f"/api/world/scene/{session_id}", timeout=self._timeout_scene)
        data = resp.json()

        present = [
            PresentCharacter(
                name=p.get("name", ""),
                role=p.get("role", ""),
                last_action=p.get("last_action", ""),
                last_seen=p.get("last_seen", ""),
            )
            for p in data.get("present", [])
        ]
        events = [
            RecentEvent(
                who=e.get("who", ""),
                summary=e.get("summary", ""),
                ts=e.get("ts", ""),
            )
            for e in data.get("recent_events_here", [])
        ]

        return SceneData(
            session_id=session_id,
            location=data.get("location", ""),
            role=data.get("role", ""),
            present=present,
            recent_events_here=events,
            location_graph=data.get("location_graph", {}),
        )

    async def get_new_events(self, session_id: str, since: str) -> list[RecentEvent]:
        """Poll for events at the agent's location since a timestamp. Fast loop trigger."""
        resp = await self._get_with_retry(
            f"/api/world/scene/{session_id}/new-events",
            params={"since": since},
            timeout=self._timeout_scene,
        )
        data = resp.json()
        return [
            RecentEvent(who=e.get("who", ""), summary=e.get("summary", ""), ts=e.get("ts", ""))
            for e in data.get("events", [])
        ]

    # ------------------------------------------------------------------
    # Actions (fast + slow loops)
    # ------------------------------------------------------------------

    async def post_action(self, session_id: str, action: str) -> TurnResult:
        """Submit a freeform action. No retries — idempotency not guaranteed."""
        resp = await self._post(
            "/api/action",
            {"session_id": session_id, "action": action},
            timeout=self._timeout_action,
        )
        data = resp.json()
        return TurnResult(
            narrative=data.get("narrative", ""),
            choices=data.get("choices", []),
            vars=data.get("vars", {}),
            plausible=data.get("plausible", True),
        )

    async def post_next(self, session_id: str, vars: dict, choice_taken: dict | None = None) -> TurnResult:
        """Advance to next storylet."""
        payload: dict[str, Any] = {"session_id": session_id, "vars": vars}
        if choice_taken:
            payload["choice_taken"] = choice_taken
        resp = await self._post("/api/next", payload, timeout=self._timeout_action)
        data = resp.json()
        return TurnResult(
            narrative=data.get("text", ""),
            choices=data.get("choices", []),
            vars=data.get("vars", {}),
        )

    # ------------------------------------------------------------------
    # World memory (slow loop context)
    # ------------------------------------------------------------------

    async def get_world_facts(self, query: str, session_id: str | None = None, limit: int = 5) -> list[WorldFact]:
        """Semantic search over world event history. Uses server-side embeddings."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if session_id:
            params["session_id"] = session_id
        resp = await self._get_with_retry("/api/world/facts", params=params, timeout=self._timeout_scene)
        data = resp.json()
        return [
            WorldFact(summary=f.get("summary", ""))
            for f in data.get("facts", [])
            if f.get("summary")
        ]

    async def get_graph_facts(self, query: str, session_id: str | None = None, limit: int = 5) -> list[WorldFact]:
        """Semantic search over active world fact graph. Uses server-side embeddings."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if session_id:
            params["session_id"] = session_id
        resp = await self._get_with_retry("/api/world/graph/facts", params=params, timeout=self._timeout_scene)
        data = resp.json()
        return [
            WorldFact(
                summary=f.get("summary", ""),
                subject=f.get("subject_node", {}).get("name", "") if isinstance(f.get("subject_node"), dict) else "",
                predicate=f.get("predicate", ""),
                value=f.get("value", ""),
                confidence=f.get("confidence", 1.0),
            )
            for f in data.get("facts", [])
            if f.get("summary")
        ]

    # ------------------------------------------------------------------
    # Letters (mail loop)
    # ------------------------------------------------------------------

    async def get_inbox(self, agent_name: str) -> list[Letter]:
        """Mail loop: poll for unread letters waiting for this agent."""
        resp = await self._get_with_retry(
            f"/api/world/letters/inbox/{agent_name}", timeout=self._timeout_scene
        )
        data = resp.json()
        return [
            Letter(filename=l.get("filename", ""), body=l.get("body", ""))
            for l in data.get("letters", [])
        ]

    async def send_letter(self, from_name: str, to_agent: str, body: str, session_id: str) -> dict:
        resp = await self._post(
            "/api/world/letter",
            {"from_name": from_name, "to_agent": to_agent, "body": body, "session_id": session_id},
            timeout=30.0,
        )
        return resp.json()

    async def reply_letter(self, from_agent: str, to_session_id: str, body: str) -> dict:
        resp = await self._post(
            "/api/world/letter/reply",
            {"from_agent": from_agent, "to_session_id": to_session_id, "body": body},
            timeout=30.0,
        )
        return resp.json()

    # ------------------------------------------------------------------
    # Location chat (co-located async messaging)
    # ------------------------------------------------------------------

    async def get_location_chat(self, location: str, since: str | None = None) -> list[ChatMessage]:
        """Return recent chat messages at a location. Used by fast loop."""
        params: dict[str, Any] = {"limit": "30"}
        if since:
            params["since"] = since
        resp = await self._get_with_retry(
            f"/api/world/location/{location}/chat",
            params=params,
            timeout=self._timeout_scene,
        )
        data = resp.json()
        return [
            ChatMessage(
                id=m.get("id", 0),
                session_id=m.get("session_id", ""),
                display_name=m.get("display_name") or m.get("session_id", "")[:12],
                message=m.get("message", ""),
                ts=m.get("ts", ""),
            )
            for m in data.get("messages", [])
        ]

    async def post_location_chat(
        self,
        location: str,
        session_id: str,
        message: str,
        display_name: str,
    ) -> dict:
        """Post a chat message at a location on behalf of an agent."""
        resp = await self._post(
            f"/api/world/location/{location}/chat",
            {"session_id": session_id, "message": message, "display_name": display_name},
            timeout=30.0,
        )
        return resp.json()

    # ------------------------------------------------------------------
    # Real-world grounding + map movement
    # ------------------------------------------------------------------

    async def get_grounding(self) -> dict:
        """
        Fetch current SF time + weather from the worldweaver grounding endpoint.
        Keys: datetime_str, day_of_week, time_of_day, season, hour, month,
              weather, temperature_f, weather_description
        Returns empty dict on failure — callers must handle gracefully.
        """
        try:
            resp = await self._get("/api/world/grounding", timeout=8.0)
            return resp.json()
        except Exception as e:
            logger.debug("[grounding] fetch failed: %s", e)
            return {}

    async def post_map_move(self, session_id: str, destination: str) -> dict:
        """
        Move one hop toward destination along the city graph.
        Bypasses NL movement detection — explicit map route.
        Returns: {moved, from_location, to_location, route, route_remaining, narrative}
        """
        resp = await self._post(
            "/api/game/move",
            {"session_id": session_id, "destination": destination},
            timeout=30.0,
        )
        return resp.json()

    # City map — grounded geography for slow loop context
    # ------------------------------------------------------------------

    async def get_location_map_context(self, session_id: str, location: str) -> str:
        """
        Fetch compressed prose geography context for a location.
        Returns a short text block (neighborhood, adjacency, transit, landmarks)
        suitable for injection into the slow loop prompt.
        Returns empty string if no city pack is available.
        """
        try:
            resp = await self._get_with_retry(
                f"/api/world/map/{session_id}/context",
                params={"location": location},
                timeout=10.0,
            )
            data = resp.json()
            return data.get("context", "")
        except Exception as e:
            logger.debug("[map] context fetch failed for %s: %s", location, e)
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, *, params: dict | None = None, timeout: float = 30.0) -> httpx.Response:
        try:
            resp = await self._client.get(path, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            raise WorldClientError(f"GET {path} returned {e.response.status_code}") from e

    async def _get_with_retry(
        self, path: str, *, params: dict | None = None, timeout: float = 30.0, max_retries: int = 2
    ) -> httpx.Response:
        retryable = {429, 500, 502, 503}
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.get(path, params=params, timeout=timeout)
                if resp.status_code in retryable:
                    delay = 2 ** attempt
                    logger.warning("world GET %s: HTTP %s, retrying in %ss", path, resp.status_code, delay)
                    await asyncio.sleep(delay)
                    last_error = WorldClientError(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp
            except httpx.TimeoutException as e:
                last_error = WorldClientError(f"GET {path} timed out")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue

        raise last_error or WorldClientError(f"GET {path} failed")

    async def _post(self, path: str, payload: dict, *, timeout: float = 60.0) -> httpx.Response:
        try:
            resp = await self._client.post(path, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            raise WorldClientError(
                f"POST {path} returned {e.response.status_code}: {e.response.text[:200]}"
            ) from e

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> WorldWeaverClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
