from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.identity.loader import IdentityLoader, ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.fast import FastLoop
from src.loops.mail import MailLoop
from src.loops.slow import SlowLoop
from src.memory.provisional import ProvisionalScratchpad
from src.memory.retrieval import LongTermMemory
from src.memory.working import WorkingMemory
from src.world.client import WorldWeaverClient

logger = logging.getLogger(__name__)


class Resident:
    """
    A single running agent: one character, three loops, shared memory.

    Residents are autonomous — they boot themselves, manage their own
    session with the world server, and run until cancelled.

    The resident doesn't know about other residents. It knows who it is
    (SOUL.md), what it's been doing (working memory), and what it's been
    noticing (provisional scratchpad). Everything else comes from the world.
    """

    def __init__(
        self,
        resident_dir: Path,
        ww_client: WorldWeaverClient,
        llm: InferenceClient,
    ):
        self._resident_dir = resident_dir
        self._ww = ww_client
        self._llm = llm
        self._identity: ResidentIdentity | None = None
        self._session_id: str | None = None
        self._tasks: list[asyncio.Task] = []

    @property
    def name(self) -> str:
        if self._identity:
            return self._identity.name
        return self._resident_dir.name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, world_id: str) -> None:
        """
        Load identity, establish session, wire up loops. Call before run().
        """
        self._identity = IdentityLoader.load(self._resident_dir)
        logger.info("[%s] identity loaded", self.name)

        self._session_id = await self._get_or_create_session(world_id)
        logger.info("[%s] session: %s", self.name, self._session_id)

    async def run(self) -> None:
        """
        Run fast, slow, and mail loops concurrently.
        Returns when all loops stop (or any raises an unhandled exception).
        """
        if not self._identity or not self._session_id:
            raise RuntimeError(f"Resident {self.name} not started — call start() first")

        identity = self._identity
        session_id = self._session_id

        # Shared memory — all three loops read/write these
        working = WorkingMemory(
            self._resident_dir / "memory" / "working.json",
            max_items=identity.tuning.slow_max_context_events,
        )
        provisional = ProvisionalScratchpad(self._resident_dir / "memory" / "impressions")
        long_term = LongTermMemory(self._resident_dir / "memory" / "long_term.json")

        fast = FastLoop(
            identity=identity,
            resident_dir=self._resident_dir,
            ww_client=self._ww,
            llm=self._llm,
            session_id=session_id,
            working_memory=working,
            provisional=provisional,
        )

        slow = SlowLoop(
            identity=identity,
            resident_dir=self._resident_dir,
            ww_client=self._ww,
            llm=self._llm,
            session_id=session_id,
            working_memory=working,
            provisional=provisional,
            long_term=long_term,
        )

        loops: list[asyncio.Coroutine] = [fast.run(), slow.run()]

        if identity.tuning.mail_enabled:
            mail = MailLoop(
                identity=identity,
                resident_dir=self._resident_dir,
                ww_client=self._ww,
                llm=self._llm,
                session_id=session_id,
            )
            loops.append(mail.run())

        logger.info("[%s] all loops starting", self.name)

        try:
            await asyncio.gather(*loops)
        except asyncio.CancelledError:
            logger.info("[%s] resident cancelled", self.name)
            raise

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_or_create_session(self, world_id: str) -> str:
        session_path = self._resident_dir / "session_id.txt"

        if session_path.exists():
            session_id = session_path.read_text(encoding="utf-8").strip()
            try:
                await self._ww.get_scene(session_id)
                logger.debug("[%s] resumed session: %s", self.name, session_id)
                return session_id
            except Exception:
                logger.info("[%s] session %s stale — creating new session", self.name, session_id)
                session_path.unlink(missing_ok=True)

        # New session — bootstrap with the world server.
        # Session ID uses slug-YYYYMMDD-HHMMSS so the digest endpoint can
        # extract the character name for display ("rowan-20260310-..." → "Rowan").
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name_slug = self._identity.name.lower()
        session_id = f"{name_slug}-{ts}"
        identity = self._identity

        # player_role format "Name — vibe" lets the server extract just the name
        player_role = f"{identity.name} — {identity.vibe}" if identity.vibe else identity.name

        entry_location_path = self._resident_dir / "identity" / "entry_location.txt"
        entry_location = ""
        if entry_location_path.exists():
            entry_location = entry_location_path.read_text(encoding="utf-8").strip()
            entry_location_path.unlink()  # consume once

        await self._ww.bootstrap_session(
            session_id=session_id,
            world_id=world_id,
            world_theme="",           # server uses existing world theme
            player_role=player_role,
            tone="natural, grounded",
            description=identity.soul[:300],
            entry_location=entry_location,
        )

        session_path.write_text(session_id, encoding="utf-8")
        logger.info("[%s] bootstrapped new session: %s", self.name, session_id)
        return session_id
