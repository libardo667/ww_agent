# src/world/ — WorldWeaver API Client

## Purpose

Async HTTP client for the WorldWeaver server. Handles all communication
between a resident and the shared world. The WorldWeaver server is the
canonical source of truth — this client is read/write but never authoritative.

## Endpoints Used

### Scene (Fast + Slow loops)

```
GET /api/world/scene/{session_id}
```
Returns: current location, colocated characters (with activity summaries and
vibes), recent local events. This is the fast loop's entire context.

### Action (Fast + Slow loops)

```
POST /api/action
{
    "session_id": "...",
    "action": "I lean against the wall and listen to the pipes."
}
```
Returns: narrative text, choices, updated vars. The reducer validates and
commits the action to the world fact graph.

### Next (Slow loop — choice selection)

```
POST /api/next
{
    "session_id": "...",
    "vars": { "visited_cafe": true }
}
```
Returns: next scene narrative, new choices, updated vars.

### Letters (Mail loop only)

```
POST /api/world/letter
{
    "from_name": "nadia",
    "to_agent": "margot",
    "body": "...",
    "session_id": "..."
}

POST /api/world/letter/reply
{
    "from_agent": "nadia",
    "to_session_id": "...",
    "body": "..."
}
```

### Scene Events (Fast loop trigger — future)

```
GET /api/world/scene/{session_id}/new-events?since={timestamp}
```
Returns events at the agent's location since the given timestamp. Used by
the fast loop to determine whether to fire or skip. When WorldWeaver adds
SSE/webhook support, this polling endpoint becomes unnecessary.

### Session Bootstrap (Resident initialization)

```
POST /api/session/bootstrap
{
    "session_id": "...",
    "world_id": "...",
    "world_theme": "...",
    "player_role": "...",
    "tone": "...",
    "storylet_count": 8,
    "bootstrap_source": "worldweaver-agent"
}
```

### Health

```
GET /health
```
Returns `{"ok": true}` if server is up. Used during startup to wait for server.

## Interface Contract

```python
class WorldWeaverClient:
    async def health(self) -> bool
    async def wait_for_ready(self, timeout_seconds: float = 120.0) -> None
    async def bootstrap_session(self, session_id: str, world_id: str, role: str, ...) -> dict
    async def get_scene(self, session_id: str) -> SceneData
    async def post_action(self, session_id: str, action: str) -> TurnResult
    async def post_next(self, session_id: str, vars: dict) -> TurnResult
    async def get_new_events(self, session_id: str, since: str) -> list[Event]
    async def send_letter(self, from_name: str, to_agent: str, body: str, session_id: str) -> dict
    async def reply_letter(self, from_agent: str, to_session_id: str, body: str) -> dict
```

## Scene Data — Prose Before LLM

`SceneData` from the API is structured data. Before it reaches any LLM
prompt, the loop should transform it into prose. The agent should never see
JSON field names, colocated character arrays, or raw choice lists.

Write a `scene_to_prose(scene: SceneData, character_name: str) -> str`
helper that produces something like:

> *You are in the Deeper Corridor. The air is heavier here. Casper is nearby
> — he set something down a few minutes ago and hasn't moved since. Elias
> left a while back.*

rather than presenting the raw `SceneData` struct. The loop calls this
helper before building the prompt. The API response schema is an
implementation detail the agent should never touch.

## Connection Model

- Single `httpx.AsyncClient` shared across all residents.
- Connection pooling handles concurrent requests from multiple loops.
- Timeout: 30s for scene reads, 120s for actions (LLM narration can be slow).
- No retries on action/next (idempotency is not guaranteed). Retry on scene reads.

## Data Types

Define simple dataclasses for API responses:

```python
@dataclass
class ColocatedCharacter:
    name: str
    last_action: str
    vibe: str

@dataclass
class SceneData:
    location: str
    narrative: str
    colocated: list[ColocatedCharacter]
    recent_events: list[str]
    choices: list[dict]
    vars: dict

@dataclass
class TurnResult:
    narrative: str
    choices: list[dict]
    vars: dict
```
