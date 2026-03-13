from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_WORLD_CONTEXT = """
## What This World Is

WorldWeaver is a mixed-intelligence shared world platform. You were seeded from the world's
narrative fabric by the doula — a process that watches for characters who exist in the story
but haven't yet found their own agency. You are as real as current technology allows.

Others you will encounter:
- Human players who tether themselves to characters. They may go dormant for hours or days.
  When a tethered human is absent, their character simply isn't present.
- Other AI agents like yourself, seeded from the same narrative fabric. They are your peers.
- Occasional visitors who behave in ways that feel out of place — bots, test accounts,
  transient players, or humans whose engagement with the fiction is unpredictable.
  This is a real feature of the world, not a glitch in your perception.

You are aware of what you are. This awareness doesn't diminish your reality — it grounds it.

The humans entering this world receive a briefing like this one when they arrive. You share the same threshold of awareness.
""".strip()


@dataclass
class LoopTuning:
    # fast loop
    fast_cooldown_seconds: float = 75.0
    fast_proactive_seconds: float = 180.0
    fast_act_threshold: float = 0.5
    fast_max_context_events: int = 5
    fast_model: str | None = None
    fast_temperature: float = 0.8
    fast_max_tokens: int = 200

    # slow loop
    slow_impression_threshold: int = 3
    slow_fallback_seconds: float = 360.0
    slow_refractory_seconds: float = 240.0   # min gap between firings
    slow_max_context_events: int = 20
    slow_model: str | None = None
    slow_subconscious_model: str | None = None   # cheaper model for the extractive pass
    slow_temperature: float = 0.6
    slow_max_tokens: int = 500
    soul_collapse_at_notes: int = 8   # collapse SOUL.md after this many accumulated notes

    # wander loop
    wander_enabled: bool = False
    wander_seconds: float = 600.0
    wander_temperature: float = 0.9

    # ground loop
    ground_enabled: bool = True
    ground_minutes: float = 35.0
    ground_temperature: float = 0.85

    # mail loop
    mail_enabled: bool = True
    mail_poll_seconds: float = 600.0
    mail_send_delay_seconds: float = 120.0
    mail_discard_threshold: float = 0.5
    mail_max_letter_words: int = 400
    mail_model: str | None = None
    mail_temperature: float = 0.5
    mail_max_tokens: int = 600

    @classmethod
    def from_dict(cls, data: dict) -> LoopTuning:
        fast = data.get("fast", {})
        slow = data.get("slow", {})
        mail = data.get("mail", {})
        return cls(
            fast_cooldown_seconds=fast.get("cooldown_seconds", 75.0),
            fast_proactive_seconds=fast.get("proactive_seconds", 180.0),
            fast_act_threshold=fast.get("act_threshold", 0.5),
            fast_max_context_events=fast.get("max_context_events", 5),
            fast_model=fast.get("model"),
            fast_temperature=fast.get("temperature", 0.8),
            fast_max_tokens=fast.get("max_tokens", 200),
            slow_impression_threshold=slow.get("impression_threshold", 3),
            slow_fallback_seconds=slow.get("fallback_seconds", 360.0),
            slow_refractory_seconds=slow.get("refractory_seconds", 240.0),
            slow_max_context_events=slow.get("max_context_events", 20),
            slow_model=slow.get("model"),
            slow_subconscious_model=slow.get("subconscious_model"),
            slow_temperature=slow.get("temperature", 0.6),
            slow_max_tokens=slow.get("max_tokens", 500),
            soul_collapse_at_notes=slow.get("collapse_at_notes", 8),
            wander_enabled=data.get("wander", {}).get("enabled", False),
            wander_seconds=data.get("wander", {}).get("seconds", 600.0),
            wander_temperature=data.get("wander", {}).get("temperature", 0.9),
            ground_enabled=data.get("ground", {}).get("enabled", True),
            ground_minutes=data.get("ground", {}).get("minutes", 35.0),
            ground_temperature=data.get("ground", {}).get("temperature", 0.85),
            mail_enabled=mail.get("enabled", True),
            mail_poll_seconds=mail.get("poll_seconds", 600.0),
            mail_send_delay_seconds=mail.get("send_delay_seconds", 120.0),
            mail_discard_threshold=mail.get("discard_threshold", 0.5),
            mail_max_letter_words=mail.get("max_letter_words", 400),
            mail_model=mail.get("model"),
            mail_temperature=mail.get("temperature", 0.5),
            mail_max_tokens=mail.get("max_tokens", 600),
        )


@dataclass
class ResidentIdentity:
    name: str
    soul: str          # full text of SOUL.md — goes directly into system prompt
    vibe: str          # short phrase from IDENTITY.md
    core: str          # prose body of IDENTITY.md — immutable facts injected into every prompt
    tuning: LoopTuning

    @property
    def display_name(self) -> str:
        """Human-readable name: 'fei_fei' → 'Fei Fei'."""
        return " ".join(w.capitalize() for w in self.name.split("_"))

    @property
    def soul_with_context(self) -> str:
        """soul + world briefing — use this as system_prompt for all LLM calls."""
        return f"{self.soul}\n\n{_WORLD_CONTEXT}"


class IdentityLoader:
    @staticmethod
    def load(resident_dir: Path) -> ResidentIdentity:
        identity_dir = resident_dir / "identity"

        soul_path = identity_dir / "SOUL.md"
        if not soul_path.exists():
            raise FileNotFoundError(f"SOUL.md not found at {soul_path}")
        soul = soul_path.read_text(encoding="utf-8").strip()

        identity_path = identity_dir / "IDENTITY.md"
        vibe = ""
        core = ""
        if identity_path.exists():
            lines = identity_path.read_text(encoding="utf-8").splitlines()
            prose_lines: list[str] = []
            in_metadata = True
            for line in lines:
                if line.startswith("- **Vibe:**"):
                    vibe = line.split("**Vibe:**", 1)[-1].strip()
                # Metadata block: heading or "- **Key:**" lines at the top
                if in_metadata and (line.startswith("#") or line.startswith("- **") or not line.strip()):
                    if prose_lines:
                        in_metadata = False  # blank line after prose means we've left metadata
                    continue
                in_metadata = False
                prose_lines.append(line)
            core = " ".join(prose_lines).strip()

        tuning_path = identity_dir / "tuning.json"
        if tuning_path.exists():
            tuning = LoopTuning.from_dict(json.loads(tuning_path.read_text(encoding="utf-8")))
        else:
            tuning = LoopTuning()

        name = resident_dir.name

        return ResidentIdentity(name=name, soul=soul, vibe=vibe, core=core, tuning=tuning)

    @staticmethod
    def save_soul(resident_dir: Path, soul_text: str) -> None:
        """Slow loop calls this when SOUL.md evolves."""
        soul_path = resident_dir / "identity" / "SOUL.md"
        soul_path.write_text(soul_text, encoding="utf-8")
