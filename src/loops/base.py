from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseLoop(ABC):
    """
    Abstract base for fast, slow, and mail loops.

    All loops follow the same rhythm:
        wait for trigger → gather context → decide → execute → cooldown

    Capability contracts are enforced by what each loop *has access to*,
    not by instructions inside a prompt. The fast loop has no mail client.
    The mail loop has no world action client. Violations are impossible.
    """

    def __init__(self, name: str, resident_dir: Path):
        self.name = name
        self.resident_dir = resident_dir
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("[%s] %s loop starting", self.name, self.__class__.__name__)

        while self._running:
            try:
                await self._wait_for_trigger()
                context = await self._gather_context()

                if await self._should_act(context):
                    await self._decide_and_execute(context)
                else:
                    logger.debug("[%s] trigger fired but nothing to act on", self.name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("[%s] loop error: %s", self.name, e)
                # Back off before retrying to avoid hammering on persistent errors
                await asyncio.sleep(10)

            await self._cooldown()

        logger.info("[%s] %s loop stopped", self.name, self.__class__.__name__)

    def stop(self) -> None:
        self._running = False

    @abstractmethod
    async def _wait_for_trigger(self) -> None:
        """Block until this loop should fire."""

    @abstractmethod
    async def _gather_context(self) -> dict:
        """Collect everything this loop needs to make a decision."""

    @abstractmethod
    async def _should_act(self, context: dict) -> bool:
        """Return False to skip this cycle (nothing to do)."""

    @abstractmethod
    async def _decide_and_execute(self, context: dict) -> None:
        """Call the LLM, parse the response, execute the action."""

    @abstractmethod
    async def _cooldown(self) -> None:
        """Minimum pause between firings."""
