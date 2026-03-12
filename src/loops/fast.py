"""
fast.py — Reflexive dispatcher loop.

The fast loop is the lizard brain: it sees what's in front of the agent and
immediately classifies what to do. One cheap LLM call produces a single slug.
The slug routes to the appropriate handler — some are free (observe, introspect),
some trigger a second targeted LLM call (react, ground), some call the world API
directly (move, chat, mail).

Slug taxonomy (1:1 with API surface):

    observe                     — do nothing; let events accumulate
    react: <intent hint>        — narrative participation (old fast loop); hint
                                  guides but doesn't constrain the action
    move: <location name>       — immediate movement via map API; writes route
                                  to active_route.json for wander loop continuity
    chat: <message>             — short utterance posted to location chat
    mail: <Name> | <intent>     — stages a letter intent for the mail loop
    ground                      — on-demand grounding (time/weather); same
                                  quality as GroundLoop but fires now
    introspect                  — signals the slow loop to fire early

The GroundLoop continues as a background timer so agents don't go hours without
grounding if the classifier never elects 'ground'. Both write to working memory
with type="grounding" — they're complementary, not exclusive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from src.identity.loader import ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.memory.provisional import ProvisionalScratchpad
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient, ChatMessage

logger = logging.getLogger(__name__)

_ROUTE_FILE = "active_route.json"
_INTROSPECT_SIGNAL = "introspect_signal"
# After posting a chat message, suppress chat-triggered firing for this long.
# Prevents the agent from talking to itself in a loop.
_CHAT_COOLDOWN_SECONDS = 120.0

# Minimum gap (seconds) between on-demand ground calls from this loop.
# The GroundLoop handles ambient grounding; we don't need to spam it.
_GROUND_COOLDOWN_SECONDS = 600.0

# Classifier system prompt — terse, role-focused, no fluff.
_CLASSIFIER_SYSTEM = (
    "You are {name}'s reflexive mind — pure instinct, no deliberation. "
    "You see what's immediately in front of you and pick an action. "
    "Reply with EXACTLY ONE LINE in one of these formats:\n\n"
    "  observe\n"
    "  react: <what you feel like doing, ≤8 words>\n"
    "  move: <exact location name from the list>\n"
    "  chat: <what you say aloud, ≤20 words>\n"
    "  mail: <Name> | <what to write about, ≤6 words>\n"
    "  ground\n"
    "  introspect\n\n"
    "Nothing else. No punctuation after. No explanation."
)


class FastLoop(BaseLoop):
    """
    Reflexive dispatcher loop. Fires on scene events (or proactive timer).

    Each cycle: classify → slug → route to handler.
    Most cycles: one cheap LLM call.
    react/ground cycles: two LLM calls (classifier + narrative).
    observe/move/chat/mail/introspect cycles: one LLM call + optional API call.
    """

    def __init__(
        self,
        identity: ResidentIdentity,
        resident_dir: Path,
        ww_client: WorldWeaverClient,
        llm: InferenceClient,
        session_id: str,
        working_memory: WorkingMemory,
        provisional: ProvisionalScratchpad,
    ):
        super().__init__(identity.name, resident_dir)
        self._identity = identity
        self._ww = ww_client
        self._llm = llm
        self._session_id = session_id
        self._working = working_memory
        self._provisional = provisional
        self._tuning = identity.tuning
        self._last_event_ts: str = datetime.now(timezone.utc).isoformat()
        self._last_chat_ts: str = datetime.now(timezone.utc).isoformat()
        self._last_ground_ts: float = 0.0
        self._chat_cooldown_until: float = 0.0  # monotonic; set after posting chat
        self._first_boot = not working_memory.has_any()
        self._route_path = resident_dir / "memory" / _ROUTE_FILE
        self._signal_path = resident_dir / "memory" / _INTROSPECT_SIGNAL

    # ------------------------------------------------------------------
    # Trigger: poll for scene events + proactive fallback
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        if self._first_boot:
            self._first_boot = False
            logger.info("[%s:fast] first boot — firing arrival action", self.name)
            return

        poll_interval = min(self._tuning.fast_cooldown_seconds, 20.0)
        proactive_seconds = self._tuning.fast_proactive_seconds
        elapsed = 0.0

        while True:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            try:
                events = await self._ww.get_new_events(self._session_id, since=self._last_event_ts)
                if events:
                    self._last_event_ts = events[-1].ts
                    return
            except Exception as e:
                logger.debug("[%s:fast] event poll failed: %s", self.name, e)

            try:
                scene = await self._ww.get_scene(self._session_id)
                if scene.location:
                    chat = await self._ww.get_location_chat(scene.location, since=self._last_chat_ts)
                    # Only trigger on messages from other sessions, and only when
                    # not in post-chat cooldown (prevents self-triggering loops).
                    others_chat = [m for m in chat if m.session_id != self._session_id]
                    if others_chat and time.monotonic() >= self._chat_cooldown_until:
                        logger.info("[%s:fast] new chat from others at %s — firing", self.name, scene.location)
                        return
            except Exception as e:
                logger.debug("[%s:fast] chat poll failed: %s", self.name, e)

            if elapsed >= proactive_seconds:
                logger.info("[%s:fast] proactive fallback — no events for %.0fs", self.name, elapsed)
                return

    # ------------------------------------------------------------------
    # Context: scene + chat + memory + route + grounding
    # ------------------------------------------------------------------

    async def _gather_context(self) -> dict:
        scene = await self._ww.get_scene(self._session_id)
        new_chat: list[ChatMessage] = []
        if scene.location:
            try:
                new_chat = await self._ww.get_location_chat(scene.location, since=self._last_chat_ts)
                if new_chat:
                    self._last_chat_ts = new_chat[-1].ts
            except Exception as e:
                logger.debug("[%s:fast] chat fetch failed: %s", self.name, e)

        # Most recent grounding from working memory
        grounding_text = ""
        for entry in reversed(self._working.recent(8)):
            if entry.get("type") == "grounding":
                grounding_text = entry.get("text", "")
                break

        # Active route
        active_route = self._load_route()

        # Adjacent location names from graph for move: validation
        graph = scene.location_graph
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        name_to_key = {n["name"]: n["key"] for n in nodes if n.get("key") and n.get("name")}
        current_key = name_to_key.get(scene.location, f"location:{scene.location.lower()}")
        adjacent_keys: set[str] = set()
        for e in edges:
            if e.get("from") == current_key:
                adjacent_keys.add(e["to"])
            elif e.get("to") == current_key:
                adjacent_keys.add(e["from"])
        key_to_name = {n["key"]: n["name"] for n in nodes if n.get("key") and n.get("name")}
        adjacent_names = [key_to_name[k] for k in adjacent_keys if k in key_to_name]

        all_location_names = [n["name"] for n in nodes if n.get("name")]

        return {
            "scene": scene,
            "new_chat": new_chat,
            "grounding_text": grounding_text,
            "active_route": active_route,
            "adjacent_names": adjacent_names,
            "all_location_names": all_location_names,
        }

    async def _should_act(self, context: dict) -> bool:
        return True

    # ------------------------------------------------------------------
    # Classify → route
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        scene = context["scene"]
        new_chat = context["new_chat"]
        grounding_text = context["grounding_text"]
        active_route = context["active_route"]
        adjacent_names = context["adjacent_names"]
        all_location_names = context["all_location_names"]

        # --- Build classifier prompt ---
        classifier_user = self._build_classifier_prompt(
            scene, new_chat, grounding_text, active_route, adjacent_names
        )
        classifier_system = _CLASSIFIER_SYSTEM.format(name=self._identity.name)

        try:
            raw = await self._llm.complete(
                system_prompt=classifier_system,
                user_prompt=classifier_user,
                model=self._tuning.fast_model,
                temperature=0.3,
                max_tokens=30,
            )
        except Exception as e:
            logger.warning("[%s:fast] classifier failed: %s", self.name, e)
            return

        slug = raw.strip().strip(".,\"'").strip()
        if not slug:
            return

        logger.debug("[%s:fast] slug: %s", self.name, slug)
        slug_lower = slug.lower()

        # --- Route ---
        if slug_lower == "observe":
            logger.debug("[%s:fast] observe — no action", self.name)

        elif slug_lower.startswith("react:"):
            hint = slug[len("react:"):].strip()
            await self._do_react(hint, scene, new_chat)

        elif slug_lower.startswith("move:"):
            dest = slug[len("move:"):].strip()
            await self._do_move(dest, scene, all_location_names)

        elif slug_lower.startswith("chat:"):
            message = slug[len("chat:"):].strip().strip('"\'')
            if message:
                await self._do_chat(message, scene)

        elif slug_lower.startswith("mail:"):
            mail_body = slug[len("mail:"):].strip()
            if "|" in mail_body:
                recipient, intent = mail_body.split("|", 1)
                await self._do_mail(recipient.strip(), intent.strip())
            else:
                await self._do_mail(mail_body.strip(), "")

        elif slug_lower == "ground":
            await self._do_ground(scene)

        elif slug_lower == "introspect":
            self._do_introspect()

        else:
            # Unrecognized — treat as a react hint
            logger.debug("[%s:fast] unrecognized slug %r, treating as react", self.name, slug)
            await self._do_react(slug, scene, new_chat)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _do_react(self, hint: str, scene, new_chat: list) -> None:
        """Narrative participation — what the old fast loop did, now guided by the hint."""
        others = [p for p in scene.present if p.name.lower() != self.name.lower()]
        present_lines = "\n".join(
            f"- {p.name}" + (f" ({p.role})" if p.role and p.role != p.name else "")
            + (f": {p.last_action}" if p.last_action else "")
            for p in others
        ) or "(no one else)"

        event_lines = ""
        if scene.recent_events_here:
            event_lines = "\n".join(f"- {e.summary}" for e in scene.recent_events_here[-3:] if e.summary)

        recent = self._working.recent(2)
        own_lines = ""
        if recent:
            own = [e.get("action", "") for e in recent if e.get("action")]
            if own:
                own_lines = "What you've been doing: " + " / ".join(own)

        # Reverie anchor — core identity facts from IDENTITY.md, prepended before
        # every action so the character remembers who they are even under drift.
        parts = []
        if self._identity.core:
            parts.append(self._identity.core)
        parts.append(f"You're at {scene.location}.")
        parts.append(f"Present:\n{present_lines}")
        if event_lines:
            parts.append(f"Recent:\n{event_lines}")
        if own_lines:
            parts.append(own_lines)
        if hint:
            parts.append(f"What you feel like doing: {hint}.")

        if new_chat:
            chat_lines = "\n".join(f"- {m.display_name}: \"{m.message}\"" for m in new_chat[-5:])
            parts.append(
                f"Chat here:\n{chat_lines}\n\n"
                "You can reply by starting your response with REPLY: followed by what you say. "
                "Or ignore it and do something else."
            )
        else:
            parts.append("Respond naturally and briefly.")

        user_prompt = "\n\n".join(parts)

        try:
            response = await self._llm.complete(
                system_prompt=self._identity.soul,
                user_prompt=user_prompt,
                model=self._tuning.fast_model,
                temperature=self._tuning.fast_temperature,
                max_tokens=self._tuning.fast_max_tokens,
            )
        except Exception as e:
            logger.warning("[%s:fast] react LLM failed: %s", self.name, e)
            return

        action = response.strip()
        if not action:
            return

        # Chat reply opt-in
        if action.upper().startswith("REPLY:"):
            reply_text = action[len("REPLY:"):].strip().strip('"\'')

            try:
                await self._ww.post_location_chat(
                    location=scene.location,
                    session_id=self._session_id,
                    message=reply_text,
                    display_name=self._identity.name,
                )
                logger.info("[%s:fast] chat reply: %s", self.name, reply_text[:80])
                # Advance past our own message and suppress re-triggering for a while
                self._last_chat_ts = datetime.now(timezone.utc).isoformat()
                self._chat_cooldown_until = time.monotonic() + _CHAT_COOLDOWN_SECONDS
            except Exception as e:
                logger.warning("[%s:fast] chat post failed: %s", self.name, e)
            return

        impression_text, action_text = self._extract_impression(action)

        try:
            result = await self._ww.post_action(self._session_id, action_text)
            logger.info("[%s:fast] reacted: %s", self.name, action_text[:80])
            self._working.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "action",
                "loop": "fast",
                "location": scene.location,
                "action": action_text,
                "narrative": result.narrative[:200] if result.narrative else "",
            })
            if impression_text or (result.narrative and self._seems_notable(result.narrative)):
                self._provisional.write_impression(
                    trigger=action_text,
                    raw_reaction=impression_text or result.narrative[:200],
                    location=scene.location,
                    colocated=[p.name for p in scene.present if p.name.lower() != self.name.lower()],
                )
        except Exception as e:
            logger.warning("[%s:fast] post_action failed: %s", self.name, e)

    async def _do_move(self, destination: str, scene, all_location_names: list[str]) -> None:
        """Immediate movement — reactive, not timer-driven."""
        # Validate destination against known graph nodes (case-insensitive)
        dest_lower = destination.lower()
        matched = next((n for n in all_location_names if n.lower() == dest_lower), None)
        if not matched:
            logger.debug("[%s:fast] move: %r not in location graph — ignoring", self.name, destination)
            return

        logger.info("[%s:fast] moving to %s", self.name, matched)
        try:
            result = await self._ww.post_map_move(self._session_id, matched)
        except Exception as e:
            logger.warning("[%s:fast] map move failed: %s", self.name, e)
            return

        if result.get("moved"):
            arrived_at = result.get("to_location", matched)
            remaining = result.get("route_remaining", [])
            logger.info("[%s:fast] moved to %s", self.name, arrived_at)
            if remaining and arrived_at.lower() != matched.lower():
                # Multi-hop: save route for wander loop to continue
                self._save_route(matched, remaining)
            else:
                # Single hop or arrived — clear any stale route
                self._clear_route()
        else:
            logger.debug("[%s:fast] move returned moved=false: %s", self.name, result)
            self._clear_route()

    async def _do_chat(self, message: str, scene) -> None:
        """Short reactive utterance posted to location chat."""
        try:
            await self._ww.post_location_chat(
                location=scene.location,
                session_id=self._session_id,
                message=message,
                display_name=self._identity.name,
            )
            logger.info("[%s:fast] chat: %s", self.name, message[:80])
            # Advance past our own message and suppress re-triggering for a while
            self._last_chat_ts = datetime.now(timezone.utc).isoformat()
            self._chat_cooldown_until = time.monotonic() + _CHAT_COOLDOWN_SECONDS
        except Exception as e:
            logger.warning("[%s:fast] chat post failed: %s", self.name, e)

    async def _do_mail(self, recipient: str, intent: str) -> None:
        """Stage a letter intent — mail loop writes the actual letter."""
        intents_dir = self.resident_dir / "letters" / "intents"
        intents_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        intent_path = intents_dir / f"intent_{ts}_{recipient}.md"
        intent_path.write_text(
            f"To: {recipient}\nStaged-At: {ts}\n\nContext:\n{intent}",
            encoding="utf-8",
        )
        logger.info("[%s:fast] mail intent staged → %s: %s", self.name, recipient, intent[:60])

    async def _do_ground(self, scene) -> None:
        """On-demand grounding — fetch time/weather and generate a naturalistic moment."""
        import time
        now_ts = time.monotonic()
        if now_ts - self._last_ground_ts < _GROUND_COOLDOWN_SECONDS:
            logger.debug("[%s:fast] ground: cooldown active, skipping", self.name)
            return

        try:
            grounding = await self._ww.get_grounding()
        except Exception as e:
            logger.warning("[%s:fast] grounding fetch failed: %s", self.name, e)
            return

        if not grounding:
            return

        datetime_str = grounding.get("datetime_str", "")
        weather_desc = grounding.get("weather_description") or grounding.get("weather", "")
        world_line = datetime_str
        if weather_desc:
            world_line += f". Weather: {weather_desc}"

        user_prompt = (
            f"You are {self._identity.name}, currently at {scene.location}.\n\n"
            f"Right now in San Francisco: {world_line}.\n\n"
            f"In one or two sentences, describe what {self._identity.name} briefly notices — "
            f"a glance at a phone, the quality of light, the temperature, a smell. "
            f"Specific and sensory. No drama."
        )

        try:
            observation = await self._llm.complete(
                system_prompt=self._identity.soul,
                user_prompt=user_prompt,
                model=self._tuning.fast_model,
                temperature=0.8,
                max_tokens=80,
            )
        except Exception as e:
            logger.warning("[%s:fast] ground LLM failed: %s", self.name, e)
            return

        observation = observation.strip()
        if observation:
            self._working.append({
                "type": "grounding",
                "text": observation,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            self._last_ground_ts = now_ts
            logger.info("[%s:fast] ground: %s", self.name, observation[:100])

    def _do_introspect(self) -> None:
        """Signal the slow loop to fire early."""
        self._signal_path.parent.mkdir(parents=True, exist_ok=True)
        self._signal_path.touch()
        logger.info("[%s:fast] introspect signal written", self.name)

    # ------------------------------------------------------------------
    # Classifier prompt
    # ------------------------------------------------------------------

    def _build_classifier_prompt(
        self,
        scene,
        new_chat: list,
        grounding_text: str,
        active_route: dict | None,
        adjacent_names: list[str],
    ) -> str:
        parts: list[str] = []

        # Where + who
        others = [p for p in scene.present if p.name.lower() != self.name.lower()]
        present_str = ", ".join(
            (p.role if p.role and p.role != p.name else p.name) for p in others
        ) or "no one"
        parts.append(f"At: {scene.location}. Present: {present_str}.")

        # Grounding
        if grounding_text:
            parts.append(f"Context: {grounding_text}")

        # Recent events
        if scene.recent_events_here:
            events = "; ".join(e.summary for e in scene.recent_events_here[-2:] if e.summary)
            if events:
                parts.append(f"Just happened: {events}")

        # New chat
        if new_chat:
            chat_str = " / ".join(f"{m.display_name}: \"{m.message}\"" for m in new_chat[-3:])
            parts.append(f"Someone spoke: {chat_str}")

        # Own recent actions
        recent = self._working.recent(2)
        own_actions = [e.get("action", "") for e in recent if e.get("action")]
        if own_actions:
            parts.append("You've been: " + " / ".join(own_actions))

        # Movement options
        if adjacent_names:
            parts.append("Can move to: " + ", ".join(adjacent_names[:6]))

        # Active route
        if active_route:
            remaining = active_route.get("remaining", [])
            dest = active_route.get("destination", "")
            parts.append(
                f"En route to {dest} ({len(remaining)} hop{'s' if len(remaining) != 1 else ''} left). "
                f"Say 'move: {remaining[0]}' to continue." if remaining else f"En route to {dest}."
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Route file helpers (shared with wander loop)
    # ------------------------------------------------------------------

    def _load_route(self) -> dict | None:
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

    # ------------------------------------------------------------------
    # Helpers (carried over)
    # ------------------------------------------------------------------

    def _extract_impression(self, text: str) -> tuple[str, str]:
        match = re.search(r'\(\*([^)]+)\)\s*$', text)
        if match:
            return match.group(1).strip(), text[:match.start()].strip()
        return "", text

    def _seems_notable(self, narrative: str) -> bool:
        notable_words = ["strange", "unexpected", "surprised", "odd", "familiar",
                         "uneasy", "wrong", "different", "changed", "recognized"]
        lower = narrative.lower()
        return any(w in lower for w in notable_words)

    async def _cooldown(self) -> None:
        await asyncio.sleep(self._tuning.fast_cooldown_seconds)
