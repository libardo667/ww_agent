# src/identity/ — Identity System

## Purpose

Load and manage a resident's persistent identity files. Identity is the
thing that survives world resets. It's not memory (which is episodic) —
it's disposition, personality, and self-concept.

## Files

### SOUL.md

The personality kernel. Loaded at startup, cached for the fast loop, fully
readable and writable by the slow loop. Contains:

- Who the character is (background, habits, personality traits)
- Core behavioral truths (how they approach the world)
- Vibe (how they come across to others)
- WorldWeaver connection info (server URL, artifact paths)

The slow loop may rewrite SOUL.md when the character evolves — a significant
event changes how they see themselves, a relationship shifts their priorities,
a trauma alters their disposition. These edits are rare and meaningful.

### IDENTITY.md

Compact metadata:
- Name
- Creature type (WorldWeaver resident)
- Vibe (short phrase)
- Emoji (signature)

Less likely to change than SOUL.md. Loaded once at startup.

### tuning.json

Loop parameters that define personality through temporal dynamics:

```json
{
    "fast": {
        "cooldown_seconds": 75,
        "act_threshold": 0.5,
        "max_context_events": 5,
        "model": null,
        "temperature": 0.8
    },
    "slow": {
        "impression_threshold": 3,
        "fallback_seconds": 360,
        "max_context_events": 20,
        "model": null,
        "temperature": 0.6
    },
    "mail": {
        "enabled": true,
        "poll_seconds": 600,
        "send_delay_seconds": 120,
        "discard_threshold": 0.5,
        "model": null,
        "temperature": 0.5
    }
}
```

## Loader Interface

```python
@dataclass
class ResidentIdentity:
    name: str
    soul: str              # full text of SOUL.md
    identity_meta: dict    # parsed IDENTITY.md fields
    tuning: LoopTuning     # parsed tuning.json

class IdentityLoader:
    @staticmethod
    def load(resident_dir: Path) -> ResidentIdentity:
        """Load all identity files from a resident directory."""

    @staticmethod
    def save_soul(resident_dir: Path, soul_text: str) -> None:
        """Slow loop calls this when SOUL.md evolves."""
```

## Character Creation

To create a new resident:

1. Copy `residents/_template/` to `residents/{name}/`
2. Edit `identity/SOUL.md` — write who this person is
3. Edit `identity/IDENTITY.md` — fill in name, vibe, emoji
4. Edit `identity/tuning.json` — adjust loop parameters for personality
5. Restart the daemon (or send SIGHUP for hot-reload, future feature)

That's it. No registration, no config files, no database entries.
The daemon discovers residents by scanning the directory.

## Identity in Prompts

SOUL.md text is used as-is in prompts — it's already written in the
character's voice, so it doesn't need transformation. But the framing
around it matters:

**Don't say:** "Here is your SOUL.md configuration."
**Don't say:** "Your personality settings are as follows."
**Just use the text.** It opens the system prompt. It *is* who they are.
No preamble, no label, no acknowledgment that it's a file.

Similarly, tuning.json parameters should never surface to the LLM. The
character doesn't know their `act_threshold` is 0.7 or their
`impression_threshold` is 3. Those parameters shape *when* they're called
upon to respond — they're temporal, not cognitive. The character just lives
at whatever pace those parameters produce.

IDENTITY.md metadata (name, vibe, emoji) is likewise for the framework's
use — populating log lines, letter headers, colocated character summaries.
The agent doesn't need to be introduced to themselves.
