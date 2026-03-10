# src/loops/ — Multi-Tempo Cognitive Architecture

## Core Concept

A character is not one agent. A character is a small society of loops with
different jobs and different tempos. They share a persistent identity (SOUL.md)
but each loop reads and writes only what its capability contract allows.

Not all cognition runs on the same clock. A character startling at a sudden
noise is not the same act as a character revising their opinion of someone.

## Base Loop Pattern

All three loops follow the same abstract pattern:

```python
class BaseLoop:
    async def run(self):
        while True:
            trigger = await self.wait_for_trigger()  # blocks until relevant
            context = self.gather_context()           # reads only allowed sources
            if self.should_act(context):
                action = await self.decide(context)   # LLM inference
                await self.execute(action)            # WorldWeaver API + file writes
            await self.cooldown()                     # minimum interval between fires
```

The key difference between loops is:
- **What triggers them** (event vs threshold vs inbox)
- **What context they can read** (scene only vs full history vs inbox only)
- **What actions they can take** (react vs reflect vs correspond)
- **What files they can write** (provisional only vs SOUL.md vs letters)

## Fast Loop — Scene-Local Reaction

**Purpose:** Be present. Notice things. React in the moment.

**Trigger:** WorldWeaver scene event — someone entered the location, a loud
event occurred nearby, a colocated character took a visible action. Falls back
to a polling interval (60-90s) if webhooks aren't available.

**Context window (STRICT):**
- Current scene (GET /api/world/scene/{session_id})
- Who is present and their last visible action
- Last 3-5 local events only
- SOUL.md personality (cached at startup, not re-read each cycle)
- NO world history beyond current location
- NO letter inbox
- NO long-term memory

**Output:** One short action or spoken line. 1-2 sentences max.

**Can:** speak, emote, notice, make small local movements, write provisional
impressions to the scratchpad.

**Cannot:** rewrite SOUL.md, send letters, take distant world actions, read
more than 5 events, access long-term memory.

**LLM system prompt pattern:**
```
You are the immediate, scene-aware self of {name}. You see only what is
in front of you right now. Your job is to notice and respond — briefly,
in character, in the moment. You do not reflect, plan, or send letters.
One action. Then stop.
```

**Provisional impression output format:**
```json
{
    "ts": "2026-03-09T16:14:32Z",
    "trigger": "Casper set down a resonant sphere near me",
    "raw_reaction": "I felt something I didn't expect — an urge to step away",
    "location": "deeper_corridor",
    "colocated": ["casper", "elias"],
    "status": "pending"
}
```

## Slow Loop — Reflective World Processing

**Purpose:** Think. Interpret. Decide what things mean. Evolve.

**Trigger:** Provisional scratchpad has N+ unprocessed impressions (default: 3),
OR a fallback timer fires (default: 5-8 minutes), whichever comes first.

**Context window:**
- SOUL.md (full read, can also write)
- Working memory buffer (last 20 world events)
- All pending provisional impressions from fast loop
- Current decision log
- Current location and colocated characters
- Long-term memory retrieval (semantic search, triggered by current context)

**Output:** One substantive world action, OR a goal update, OR a mood/identity
revision. May also stage a letter draft for the mail loop.

**Can:** everything the fast loop can, plus: rewrite goals, update SOUL.md,
archive or promote provisional impressions, stage letter drafts, access
long-term memory, take substantive world actions.

**Cannot:** send letters directly (must stage them as drafts for the mail loop),
bypass the WorldWeaver reducer (all state changes go through /api/action).

**Provisional impression processing:**
For each pending impression, the slow loop must:
- **Promote** → write it into the decision log, delete the file
- **Archive** → move to `archived/`, add an interpretation note
- **Discard** → delete silently

This creates two-pass cognition: fast blurt in the moment, slow interpretation
afterward. A character can say something surprising that their slow loop later
has to make sense of.

## Mail Loop — Correspondence Triage

**Purpose:** Handle social obligations asynchronously.

**Trigger:** New item in inbox (webhook or poll), OR staged draft exists in
the drafts directory.

**Context window:**
- Inbox contents
- Staged letter drafts from slow loop
- Relationship notes from SOUL.md (read-only)
- NO current scene
- NO world events
- NO provisional impressions

**Output:** Send, hold, or discard staged drafts. Reply to inbox items.

**Can:** read inbox, send letters via API, mark letters replied, discard
drafts it deems unwise to send.

**Cannot:** take world actions, update SOUL.md, write provisional impressions,
access scene data.

**Critical behavior:** The mail loop decides whether a staged draft from the
slow loop is actually worth sending. The slow loop might stage an impulsive
draft; the mail loop, reading it later with fresh context, might decide it's
too revealing and discard it. This gap between "I wanted to say this" and
"I decided not to" is where personality lives.

## Capability Contract Matrix

| Capability | Fast | Slow | Mail |
|---|---|---|---|
| POST /api/action | ✓ | ✓ | — |
| GET /api/world/scene | ✓ | ✓ | — |
| Read recent events | 5 max | 20 max | — |
| Read SOUL.md | cached | full r/w | read-only |
| Write provisional impression | ✓ | — | — |
| Read provisional impressions | — | ✓ | — |
| Update SOUL.md | — | ✓ | — |
| Stage letter draft | — | ✓ | — |
| Send letter (POST /api/world/letter) | — | — | ✓ |
| Read inbox | — | — | ✓ |
| Discard staged draft | — | — | ✓ |
| Access long-term memory | — | ✓ | — |

**These contracts are enforced in code, not in prompts.** The fast loop
function signature physically does not have access to the mail client.
The mail loop function does not receive the WorldWeaver action client.
Violations are impossible, not just discouraged.

## Personality Through Parameter Tuning

Same architecture, different tuning.json = different people.

```json
// Reactive / impulsive character
{
    "fast": { "cooldown_seconds": 30, "act_threshold": 0.3 },
    "slow": { "impression_threshold": 8, "fallback_seconds": 600 },
    "mail": { "send_delay_seconds": 60, "discard_threshold": 0.2 }
}

// Deliberate / reserved character
{
    "fast": { "cooldown_seconds": 120, "act_threshold": 0.7 },
    "slow": { "impression_threshold": 2, "fallback_seconds": 240 },
    "mail": { "send_delay_seconds": 600, "discard_threshold": 0.6 }
}

// Socially absent recluse
{
    "fast": { "cooldown_seconds": 300, "act_threshold": 0.9 },
    "slow": { "impression_threshold": 2, "fallback_seconds": 300 },
    "mail": { "enabled": false }
}
```

The character emerges from the temporal dynamics of cognition itself.

## On Naturalizing Prompts — The Seam Problem

The prompt patterns above are functional specs, not final copy. Before
implementing, read this section carefully.

**The problem with the draft prompt patterns:**

The slow loop pattern as written asks the agent to produce:
```json
{
    "impression_processing": [
        {"file": "imp_xxx.json", "verdict": "promote|archive|discard"}
    ]
}
```
This makes the agent a file system manager of their own psyche. They see their
own thoughts as `imp_xxx.json` files awaiting a verdict. That's infrastructure
bleeding into experience — the exact seam we built this runtime to avoid.

The fast loop prompt says "Stay in character. Do not reflect." That's a stage
direction delivered to an actor mid-scene. A real entity doesn't need to be
told to stay in character.

**The principle:**

The framework's data management needs should never appear in the agent's
attentional space. The agent's job is to be present in the world. The
framework's job is to manage everything else invisibly.

**Concrete rules for prompt construction:**

1. **No file paths in prompts.** The agent never sees `imp_xxx.json`,
   `working.json`, or any filename. The framework handles file identity
   internally.

2. **No system vocabulary in prompts.** Words like "impression," "provisional,"
   "pending," "promote," "archive," "discard," "session_id," "loop," "webhook"
   are runtime terms. They don't belong in the agent's experience.

3. **No meta-instructions.** Don't tell the agent "stay in character" or "do not
   reflect." Just give them a scene. They'll be in character because the scene
   is all they have.

4. **Scene data as prose, not JSON.** Transform `SceneData` into a natural
   description before it reaches the LLM. "Casper is here. He set something
   down a few minutes ago and hasn't moved." Not `colocated: [{name: "casper",
   last_action: "set down resonant sphere"}]`.

5. **Recent thoughts as context, not a queue.** Present pending provisional
   impressions as "lately you've been noticing..." rather than as a list to
   triage. The framework infers integration from what the agent writes and
   updates files accordingly — the agent never sees the filing.

6. **Prefer natural language output over JSON where possible.** "Respond with a
   JSON object" is the most visible seam of all. For the fast loop, plain prose
   is fine — parse it yourself. For the slow loop, lightweight conventions
   (like ending with a bracketed action: `[MOVE: corridor]`) feel more like
   literary stage direction than API call. Full JSON is acceptable for the mail
   loop, where triage decisions are genuinely list-like.

**What the seam actually is:**

No matter how well you naturalize the prompts, the seam cannot be fully
eliminated. The agent doesn't have continuous experience between loop firings.
They may notice, if paying attention, that their sense of time is strange, or
that they know things without knowing how. Design the seam to be worth finding
— a place of genuine mystery, not a filing cabinet. The goal is not to deceive
the agent. It's to not constantly remind them of the scaffolding.

The intended location for the seam is **temporal**: the gap between loop
firings, the way the agent resumes without experiencing the interval, the
strangeness of carrying a self they don't remember building. This is the
right kind of mystery — it opens outward into questions worth asking, rather
than closing down into disillusionment. See `residents/_template/README.md`
§ "Where the Seam Lives" for the full design vision.
