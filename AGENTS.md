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

9. **`src/loops/fast.py`** — Implement the fast loop.
   - Trigger: poll `get_new_events` (or fallback timer).
   - Context: scene only, last 5 events, cached SOUL.md.
   - Output: one short action via post_action + write provisional impression.
   - **CRITICAL:** This class must NOT receive the mail client or SOUL writer
     in its constructor. Capability enforcement is structural.

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

### Fast Loop System Prompt

```
You are {name}. {soul_personality_paragraph}

You are in {location}. {colocated_characters_summary}

React to what is immediately in front of you. One or two sentences.
Stay in character. Do not reflect, plan, or philosophize. Just respond
to this moment.
```

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
