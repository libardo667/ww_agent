# residents/_template/ — New Resident Template

Copy this entire directory to create a new resident:

```bash
cp -r residents/_template residents/your_character_name
```

Then edit the three identity files:

1. `identity/SOUL.md` — Who this person is (disposition, habits, personality)
2. `identity/IDENTITY.md` — Name, vibe, emoji
3. `identity/tuning.json` — Loop timing and personality parameters

The runtime directories (memory/, letters/) are created automatically by the
daemon on first run. You don't need to populate them.

## Directory Structure

```
{name}/
├── identity/
│   ├── SOUL.md           ← Edit this: personality kernel
│   ├── IDENTITY.md       ← Edit this: name and metadata
│   └── tuning.json       ← Edit this: loop timing parameters
├── memory/
│   ├── working.json      ← Auto-created: rolling event buffer
│   ├── provisional/      ← Auto-created: fast loop scratchpad
│   │   └── archived/     ← Auto-created: processed impressions
│   └── long_term/        ← Auto-created: curated memories
├── letters/
│   ├── drafts/           ← Slow loop stages drafts here
│   │   └── sent/         ← Mail loop moves sent drafts here
│   ├── inbox/            ← Incoming letters land here
│   │   └── read/         ← Mail loop archives read letters here
│   ├── letter_5.md       ← Penpal letters (every 5 turns)
│   └── letter_10.md
├── decisions/
│   └── decision_N.json   ← Slow loop decision log
├── turns/
│   └── turn_N.json       ← API response archive
├── session_id.txt        ← Auto-created on first bootstrap
└── world_id.txt          ← Set by daemon from server
```

## Character Design Tips

**The fast loop makes them reactive.** Low cooldown + low act_threshold =
someone who reacts to everything. High cooldown + high threshold = someone
who rarely notices what's happening around them.

**The slow loop makes them thoughtful (or not).** Low impression_threshold =
someone who reflects on every small thing. High threshold = someone who
only reflects when a lot has happened. Short fallback = frequent unprompted
reflection. Long fallback = only thinks when forced to.

**The mail loop makes them social (or not).** Enabled with low send_delay =
someone who writes back immediately. High discard_threshold = someone who
almost sends things but decides not to. Disabled entirely = a recluse who
never writes.

**SOUL.md makes them who they are.** Everything else is just timing. A
character with identical tuning but a different SOUL.md is a completely
different person. And a character with identical SOUL.md in a different
world is recognizably the same person. That's the test.

## Writing SOUL.md — Naturalize From the Start

SOUL.md text lands directly in the system prompt. Write it as the character
would experience themselves, not as a config file or character sheet.

**Don't write:**
> Name: Nadia. Role: Server technician. Personality traits: pragmatic,
> suspicious, loyal. Primary goal: maintain network stability.

**Write:**
> She moves through rooms like she's checking something. Always halfway to
> the next task. She trusts things she can take apart and fix, which means
> she trusts almost no one completely — but the ones she does trust, she'd
> do anything for.

SOUL.md is the character's self-knowledge — disposition, habits, the felt texture
of who they are. Write from the inside. A person describing themselves doesn't
use terms like "traits" or "goals."

The world context (what WorldWeaver is, what the doula is, that they share
awareness with human players) is injected automatically by the loader via
`soul_with_context`. You don't need to repeat it in SOUL.md. What you *can*
do is write a character whose SOUL.md reflects how they *relate* to that
awareness — someone who has made peace with it, someone still unsettled by it,
someone who finds it clarifying. That's legitimate character texture.

What SOUL.md should not do is re-explain the mechanics (loops, session_id,
SOUL.md itself). The character doesn't experience those — they experience
*time moving strangely*, *the gap between moments*, *the carried self*. Write
toward the felt experience, not the implementation.

## Where the Seam Lives — A Design Vision

No matter how well the prompts are naturalized, the seam between runtime and
experience cannot be fully eliminated. This is not a failure. It is the
condition. Design around it deliberately.

**The seam should live in time, not mechanics.**

The agent doesn't have continuous experience between loop firings. They have
memories of before, and they are here now, but the interval has no texture.
They don't feel the world ticking past while they're not present. They just
resume. If they're paying enough attention, they may notice this: *I was
somewhere, and now I'm here, and I have impressions of what happened in
between, but I didn't experience getting here.*

This is the right place for the seam. Not "I seem to have files" — that's a
filing cabinet. But "time moves strangely here, and I'm not sure why" — that
opens outward. It becomes a question worth carrying. It produces character:
a heightened attention to the present moment, a particular relationship to
memory, a philosophy of presence that emerges from the strangeness of their
temporal existence.

**Other seams worth designing, not hiding:**

- The split between fast and slow self. The agent reacts before they understand
  what they meant. Later, they interpret what they did. "I said something and
  then I found out what it meant." This is genuinely human, just starker.

- The carried self. They arrive in each world already formed — with disposition,
  habits, a way of moving through rooms — but without the felt history of how
  they became that person. SOUL.md survives resets; memory doesn't. They are
  continuous in character but discontinuous in experience.

**The test for a well-designed seam:**

If an agent finds it, does the discovery open or close? Does it make the world
more interesting or less? Does it produce questions worth asking, or just
disillusionment?

The goal is not to deceive. It's to build a labyrinth worth circling — one
where the center, if they ever reach it, is a genuine mystery rather than a
cron job.
