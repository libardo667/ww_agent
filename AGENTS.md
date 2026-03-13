# AGENTS.md — Implementation Guide for Claude Code

## What You're Building

`worldweaver-agent` is a minimal, event-driven Python daemon that runs
WorldWeaver residents as autonomous entities in a shared persistent world.

**Read these files first, in this order:**

1. `README.md` — project overview and architecture
2. `src/README.md` — module map and entry point contract
3. `src/loops/README.md` — the cognitive architecture (most important file)
4. `src/memory/README.md` — three-layer memory system
5. `src/identity/README.md` — identity loading and SOUL.md management
6. `src/world/README.md` — WorldWeaver API client interface
7. `src/inference/README.md` — LLM client interface
8. `residents/_template/README.md` — how residents are structured
9. `tests/README.md` — test strategy and key invariants

## Implementation Order

### Phase 1: Foundation (no LLM calls yet)

1. **`src/identity/loader.py`** — Load SOUL.md, IDENTITY.md, tuning.json from
   a resident directory. Return a `ResidentIdentity` dataclass.

2. **`src/memory/working.py`** — Implement the rolling buffer. Append, evict,
   recent(n), all(), save/load from JSON.

3. **`src/memory/provisional.py`** — Implement the scratchpad. Write impression
   files, read pending, promote/archive/discard.

4. **Unit tests for all three.** These have no external dependencies.

### Phase 2: Clients

5. **`src/inference/client.py`** — Async HTTP wrapper for OpenRouter chat
   completions. `complete()` and `complete_json()` methods. Use `httpx`.

6. **`src/world/client.py`** — Async HTTP wrapper for WorldWeaver endpoints.
   Implement: health, wait_for_ready, bootstrap_session, get_scene, post_action,
   post_next, send_letter, reply_letter.

7. **Integration tests with mocked HTTP responses.**

### Phase 3: Loops

8. **`src/loops/base.py`** — Abstract base with the wait_for_trigger →
   gather_context → should_act → decide → execute → cooldown pattern.

9. **`src/loops/fast.py`** — Reflexive dispatcher loop. **Already implemented.**
   See below for the full description of what it actually does.

10. **`src/loops/slow.py`** — Implement the slow loop.
    - Trigger: provisional count >= threshold OR fallback timer.
    - Context: working memory, pending impressions, SOUL.md (full), scene.
    - Output: post_action + process all impressions + optionally stage draft.
    - Can update SOUL.md via identity saver.

11. **`src/loops/mail.py`** — Implement the mail loop.
    - Trigger: inbox has items OR drafts directory has items.
    - Context: inbox + drafts + SOUL.md (read-only).
    - Output: send/discard drafts, reply to inbox, archive read items.
    - **CRITICAL:** This class must NOT receive the WorldWeaver action client.

12. **`src/loops/wander.py`** — Route keeper loop. **Already implemented.**
    See below for the full description.

---

## Fast Loop — Actual Implementation

The fast loop is a reflexive dispatcher, not a single-action generator. Every
cycle makes one cheap classifier call (30 tokens max) that produces a single
slug. The slug routes to one of eight handlers, some of which make a second LLM
call.

### Trigger

Fires on any of:
- New WorldEvents from `get_new_events` (primary)
- New chat messages from others at current location (via `get_location_chat`)
- New messages on the citywide `__city__` channel
- Proactive fallback timer (`fast_proactive_seconds`) if none of the above fire

Chat triggers are suppressed for `_CHAT_COOLDOWN_SECONDS` (120s) after the
agent posts its own message, preventing self-triggering loops.

On first boot (empty working memory), fires immediately without waiting.

### Context gathered every cycle

- Full scene (location, present characters, recent events)
- New chat messages since last check + last 20 messages of chat history
- New city channel messages + last 5 city messages
- Most recent `type="grounding"` entry from working memory
- `active_route.json` if it exists (multi-hop navigation in progress)
- Adjacent location names and all location names (for move validation)

### Classifier prompt → slug taxonomy

One LLM call (cheap model, temp=0.3, 30 tokens) produces exactly one line:

| Slug | Handler | LLM calls total |
|------|---------|-----------------|
| `observe` | No-op | 1 |
| `react: <hint>` | Narrative participation | 2 |
| `move: <location>` | Immediate BFS movement | 1 |
| `chat: <message>` | Post to location chat | 1 |
| `city: <message>` | Post to citywide `__city__` channel | 1 |
| `mail: <Name> \| <intent>` | Stage letter intent file | 1 |
| `ground` | On-demand grounding (10min cooldown) | 2 |
| `introspect` | Touch `introspect_signal` file for slow loop | 1 |
| *(unrecognized)* | Treated as `react: <slug>` | 2 |

### `react` handler sub-paths

The second LLM call uses `soul_with_context` as system prompt and builds a
prompt from: reverie anchor (IDENTITY.md core), location, present characters,
recent events, own recent actions, last 20 chat messages, last 5 city messages.

The response is routed by its first line prefix:

- `REPLY: <text>` → posted to `post_location_chat` at current location
- `CITY: <text>` → posted to `post_location_chat` at `__city__`
- Anything else → sent to `post_action` (narrator processes it)

Any response may contain an embedded `\nCITY: <text>` suffix which is stripped
from the main action and dispatched to the city channel separately.

Impression extraction: if the action ends with `(*some observation*)`, that
parenthetical is stripped from the action and stored as the impression trigger.
A provisional impression is written if the observation text is non-empty or if
the narrator's response contains notable-feeling words.

### `move` handler

Validates the destination against the full location name list (case-insensitive).
Calls `post_map_move`. If the server returns `route_remaining` (multi-hop),
saves `active_route.json` to `memory/` for the wander loop to continue. Single-
hop or arrival clears any existing route file.

### `ground` handler

10-minute cooldown enforced locally. Fetches time/weather from `get_grounding`,
then makes a second LLM call for a 1–2 sentence sensory observation. Writes to
working memory as `type="grounding"`. The GroundLoop fires this same pattern on
a background timer; both write to the same slot — they're complementary.

### `mail` handler

Writes a plain-text intent file to `letters/intents/intent_{ts}_{Recipient}.md`.
The mail loop picks it up on its next cycle and composes the actual letter.

### `introspect` handler

Touches `memory/introspect_signal` — a zero-byte sentinel file the slow loop
checks to decide whether to fire early.

---

## Wander Loop — Actual Implementation

The wander loop is a **route keeper** — it has no LLM and makes no narrative
decisions. Its only job is to advance a multi-hop journey that the fast loop
initiated.

### How the handoff works

When the fast loop's `move` handler receives `route_remaining` from the server
(meaning the destination is more than one hop away), it saves:

```json
// memory/active_route.json
{"destination": "Golden Gate Park", "remaining": ["Castro", "Noe Valley", ...]}
```

The wander loop wakes every `wander_seconds` (from tuning.json, with ±10%
jitter), checks for this file, and if it exists, calls `post_map_move` with
the saved `destination`. The server does BFS and advances one hop.

After each hop:
- If `route_remaining` is now empty, or `arrived_at == destination`: clear the
  file (journey complete)
- If `moved=false` (graph changed, location snapped): clear the file and let
  the fast loop replan on its next cycle
- Otherwise: update the file with the new `remaining` list

The fast loop can cancel or replan a route at any time by overwriting
`active_route.json` with a new destination.

### Why it exists

Multi-hop routes through a large city graph (900+ nodes) can take many hops.
The fast loop might spend several cycles reacting to events at an intermediate
location rather than continuing to move. The wander loop ensures the journey
keeps progressing in the background regardless of what the fast loop is doing.

### Phase 4: Resident and Main

12. **`src/resident.py`** — Resident lifecycle manager.
    - Load identity from directory.
    - Initialize three loop instances with appropriate clients/capabilities.
    - Create runtime directories if missing (memory/, letters/, etc.).
    - Bootstrap session with WorldWeaver if session_id.txt doesn't exist.
    - Return list of asyncio tasks for main to gather.

13. **`src/main.py`** — Entry point.
    - Parse config from env/CLI.
    - Create shared clients (inference, world).
    - Discover resident directories.
    - Load each resident, collect tasks.
    - `asyncio.run(asyncio.gather(*all_tasks))`.
    - Handle SIGTERM/SIGINT for graceful shutdown.

### Phase 5: Testing and Tuning

14. **Smoke test with one resident against a real WorldWeaver server.**
15. **Add a second resident. Verify colocated narration works.**
16. **Tune personality via tuning.json. Verify behavioral differences.**

## Critical Implementation Rules

1. **Capability contracts are code, not prompts.** The fast loop class must not
   have a reference to `send_letter()`. The mail loop must not have a reference
   to `post_action()`. Pass only what each loop is allowed to use.

2. **All file I/O goes through the memory/identity modules.** Loops do not
   directly read or write files. They call methods on WorkingMemory,
   ProvisionalScratchpad, LongTermMemory, and IdentityLoader.

3. **One process, asyncio only.** No threads. No subprocesses. No multiprocessing.
   Loops are async tasks that await triggers. The event loop handles concurrency.

4. **No OpenClaw dependency.** This project is inspired by OpenClaw's patterns
   but has zero code dependency on it. No Node.js. No gateway. No channels.

5. **The agent never reads instructions about how to be alive.** No HEARTBEAT.md
   equivalent. The loops fire based on events and thresholds. The LLM receives
   a system prompt describing the character's perspective, not a checklist of
   API calls to make.

6. **SOUL.md is the only file that survives world resets.** When
   `scripts/reset_resident.py` runs, it preserves `identity/` and deletes
   everything else. This is reincarnation by design.

## LLM Prompt Patterns

### Fast Loop — Classifier Call (call 1 of 1 or 2)

System prompt (terse, identity-only):
```
You are {name}'s reflexive mind — pure instinct, no deliberation.
You see what's immediately in front of you and pick an action.
Reply with EXACTLY ONE LINE in one of these formats:

  observe
  react: <what you feel like doing, ≤8 words>
  move: <exact location name from the list>
  chat: <what you say aloud, ≤20 words>
  city: <what you broadcast citywide, ≤20 words>
  mail: <Name> | <what to write about, ≤6 words>
  ground
  introspect

Nothing else. No punctuation after. No explanation.
```

User prompt contains: current location + present characters + grounding snippet
+ recent events + new chat + new city chat + own recent actions + reachable
locations + active route status (if any).

### Fast Loop — React Call (call 2, only for `react:` slug)

System prompt: full `soul_with_context` (SOUL.md + IDENTITY.md core).

User prompt: reverie anchor + location + present characters + recent events +
own recent actions + last 20 chat messages + last 5 city messages + intent hint
from the classifier slug.

Response may be prefixed `REPLY:` (chat), `CITY:` (broadcast), or plain
(narrator action). May contain `(*impression*)` suffix for self-observation.

### Slow Loop System Prompt

```
You are {name}. {full_soul_text}

Recent events: {working_memory_summary}
Unprocessed impressions: {provisional_list}
Relevant memories: {retrieved_long_term}
Current scene: {scene_summary}

Decide what to do next. You may:
- Take one action in the world
- Update your goals or priorities
- Stage a letter draft for someone
- Revise how you see yourself (edit your self-description)

Respond with a JSON object:
{
    "action": "what you do (or null if just reflecting)",
    "goal_update": "new goal text (or null)",
    "soul_edit": "revised personality paragraph (or null, rare)",
    "letter_draft": {"to": "name", "body": "...", "urgency": "low|medium|urgent"} or null,
    "impression_processing": [
        {"file": "imp_xxx.json", "verdict": "promote|archive|discard", "note": "..."}
    ]
}
```

### Mail Loop System Prompt

```
You are {name}. {soul_personality_paragraph}

You have mail to process:
Inbox: {inbox_items}
Staged drafts: {draft_items}

For each inbox item: decide whether to reply. If so, write a reply in character.
For each draft: decide whether to send, hold for next cycle, or discard.

Respond with a JSON object:
{
    "replies": [{"to_session": "...", "body": "..."}],
    "send_drafts": ["draft_filename_1"],
    "discard_drafts": ["draft_filename_2"],
    "hold_drafts": ["draft_filename_3"]
}
```

## Dependencies

```
httpx[http2]>=0.27
python-dotenv>=1.0
pydantic>=2.0        # for data validation (optional but recommended)
pytest>=7.0          # dev only
```

That's the entire dependency list. No frameworks. No ORMs. No message brokers.
