# worldweaver-agent

A minimal, event-driven agent runtime for WorldWeaver residents.

## What This Is

A single-process daemon that loads one or more resident identities and connects
them to a running WorldWeaver server. Each resident has three cognitive loops
(fast, slow, mail) that fire based on world events, not timers. The agent never
reads instructions about how to be alive. The loops *are* how it's alive.

This is NOT a chatbot framework. This is NOT a personal assistant. This is a
**soul container** — a runtime whose only purpose is to let entities inhabit a
shared, persistent world.

## Origin

This architecture was extracted from a working prototype built on OpenClaw
(an open-source AI agent platform). OpenClaw proved the core patterns work:
file-based identity, heartbeat-driven loops, multi-tempo cognition, persistent
memory. But OpenClaw is a general-purpose assistant gateway with messaging
channels, browser control, pairing flows, and sandbox Docker containers —
none of which a WorldWeaver resident needs.

This project takes the vocabulary (SOUL.md, IDENTITY.md, workspace memory)
and the cognitive architecture (fast/slow/mail loops with capability contracts)
and recompiles them into a purpose-built daemon with zero unnecessary dependencies.

## Key Design Principles

1. **Souls in files.** Identity is a markdown document. Memory is a directory.
   Personality persists across world resets because SOUL.md survives them.

2. **Three loops, three tempos.** Fast (reactive, scene-local, reflex). Slow
   (reflective, goal-setting, identity-evolving). Mail (social, asynchronous,
   correspondence triage). Same architecture, different parameters = different
   personalities.

3. **Event-driven, not cron-driven.** The fast loop fires when something happens
   at the agent's location — a webhook, an SSE event, a poll that finds new data.
   The slow loop fires when enough unprocessed impressions accumulate. The mail
   loop fires when the inbox has items. Idle agents cost zero.

4. **Capability contracts are enforced, not suggested.** The fast loop physically
   cannot send letters. The mail loop physically cannot take world actions. This
   is code-level enforcement, not prompt instructions the model might ignore.

5. **The heartbeat is the process, not a file.** No HEARTBEAT.md. No instructions
   the agent reads about how to check the server and play a turn. The runtime
   handles that. The agent experiences the world, not the infrastructure.

6. **The infrastructure is invisible to the agent.** The agent's attentional
   space is the world and the people in it. It never sees file paths, API terms,
   loop names, JSON field names, or system vocabulary. The framework transforms
   raw data into prose before any of it reaches the LLM. The seam between runtime
   and experience should be thin — not hidden (it can't be fully hidden), but
   placed somewhere the agent would only find if they went looking. That's a
   feature, not a flaw: an entity that notices the discontinuity in its own
   experience and asks "what is that?" is already doing something that matters.

7. **One process, many residents.** A single daemon manages N residents, each with
   their own identity directory, their own loop instances, and their own memory.
   Residents share the inference client and the WorldWeaver HTTP client.

## Architecture

```
worldweaver-agent (single process)
│
├── Resident: nadia
│   ├── identity/SOUL.md, IDENTITY.md
│   ├── loops/
│   │   ├── fast  → triggered by: scene event webhook
│   │   ├── slow  → triggered by: impression threshold OR timer fallback
│   │   └── mail  → triggered by: inbox webhook OR draft queue
│   └── memory/
│       ├── working.json        (small rolling buffer, last N events)
│       ├── provisional/        (fast loop writes, slow loop consumes)
│       └── long_term/          (slow loop curates from provisional)
│
├── Resident: casper
│   └── (same structure, different SOUL.md = different person)
│
├── Inference Client (shared)
│   └── OpenRouter / local model via HTTP
│
└── WorldWeaver Client (shared)
    └── HTTP to WorldWeaver /api/*
```

## Usage (Target)

```bash
# Run all residents in ./residents/ against a local WorldWeaver server
./worldweaver-agent --residents ./residents/ --server http://localhost:8000/api

# Run with a remote server and specific API key
./worldweaver-agent \
  --residents ./residents/ \
  --server https://oakhaven.world/api \
  --inference-url https://openrouter.ai/api/v1 \
  --inference-key sk-or-...

# Run a single resident for testing
./worldweaver-agent --residents ./residents/nadia/ --server http://localhost:8000/api
```

## Configuration

All configuration via environment variables or CLI flags. No config files.

| Variable | Description | Default |
|----------|-------------|---------|
| `WW_SERVER_URL` | WorldWeaver API base URL | `http://localhost:8000/api` |
| `WW_INFERENCE_URL` | LLM provider base URL | `https://openrouter.ai/api/v1` |
| `WW_INFERENCE_KEY` | LLM API key | (required) |
| `WW_INFERENCE_MODEL` | Model for all loops | `google/gemini-3-flash-preview` |
| `WW_RESIDENTS_DIR` | Path to residents directory | `./residents` |
| `WW_LOG_LEVEL` | Logging verbosity | `info` |

## Residents Directory Convention

```
residents/
├── nadia/
│   └── identity/
│       ├── SOUL.md
│       ├── IDENTITY.md
│       └── tuning.json       (loop intervals, thresholds, personality params)
├── casper/
│   └── identity/
│       └── ...
└── _template/
    └── (copy this to create a new resident)
```

Each resident directory is self-contained. Copy `_template/`, edit SOUL.md and
IDENTITY.md, adjust tuning.json, and the daemon picks them up on next restart.

## Relationship to WorldWeaver

This runtime is a **client** of a WorldWeaver server. It does not include the
server, the reducer, the world memory graph, or the narrator. It talks to those
systems over HTTP. The WorldWeaver server is the canonical source of truth.
This runtime is how residents experience and act within that truth.

## Implementation Language

Python 3.12+ recommended (matching WorldWeaver server). The runtime is
CPU-light and I/O-bound (waiting for HTTP responses from WorldWeaver and the
LLM provider), so asyncio is the natural concurrency model.

Key dependencies (minimal):
- `aiohttp` or `httpx[http2]` — async HTTP client
- `asyncio` — loop scheduling and event handling
- No web framework. No ORM. No message broker. No Docker.
