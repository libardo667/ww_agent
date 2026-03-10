from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.identity.loader import ResidentIdentity
from src.inference.client import InferenceClient
from src.loops.base import BaseLoop
from src.world.client import Letter, WorldWeaverClient

logger = logging.getLogger(__name__)

_RE_REPLY   = re.compile(r'\[REPLY TO:\s*(.+?)\s*\|\s*(.+?)\]', re.IGNORECASE | re.DOTALL)
_RE_SEND    = re.compile(r'\[SEND:\s*(.+?)\]', re.IGNORECASE)
_RE_HOLD    = re.compile(r'\[HOLD:\s*(.+?)\]', re.IGNORECASE)
_RE_DISCARD = re.compile(r'\[DISCARD:\s*(.+?)\]', re.IGNORECASE)


class MailLoop(BaseLoop):
    """
    Correspondence loop. Fires when the inbox has letters or drafts are staged.

    The agent receives their correspondence as prose — letters as they'd
    read them, drafts as they'd re-read them before sending. They respond
    naturally. Light bracketed tags let the framework act on their decisions.

    This loop has no access to world actions or soul editing.
    Capability is enforced by what it has — not by rules in a prompt.
    """

    def __init__(
        self,
        identity: ResidentIdentity,
        resident_dir: Path,
        ww_client: WorldWeaverClient,
        llm: InferenceClient,
        session_id: str,
    ):
        super().__init__(identity.name, resident_dir)
        self._identity = identity
        self._ww = ww_client
        self._llm = llm
        self._session_id = session_id
        self._tuning = identity.tuning
        self._drafts_dir = resident_dir / "letters" / "drafts"
        self._sent_dir = resident_dir / "letters" / "drafts" / "sent"
        self._inbox_dir = resident_dir / "letters" / "inbox"
        self._read_dir = resident_dir / "letters" / "inbox" / "read"
        for d in [self._drafts_dir, self._sent_dir, self._inbox_dir, self._read_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Trigger: inbox has items OR staged drafts exist
    # ------------------------------------------------------------------

    async def _wait_for_trigger(self) -> None:
        while True:
            await asyncio.sleep(self._tuning.mail_poll_seconds)
            if await self._has_work():
                return

    async def _has_work(self) -> bool:
        # Check server inbox
        try:
            letters = await self._ww.get_inbox(self.name)
            if letters:
                return True
        except Exception:
            pass

        # Check local staged drafts
        drafts = list(self._drafts_dir.glob("draft_*.md"))
        return len(drafts) > 0

    async def _gather_context(self) -> dict:
        letters: list[Letter] = []
        try:
            letters = await self._ww.get_inbox(self.name)
        except Exception as e:
            logger.debug("[%s:mail] inbox fetch failed: %s", self.name, e)

        drafts = list(self._drafts_dir.glob("draft_*.md"))

        return {"letters": letters, "drafts": drafts}

    async def _should_act(self, context: dict) -> bool:
        return bool(context["letters"] or context["drafts"])

    # ------------------------------------------------------------------
    # Decide and execute
    # ------------------------------------------------------------------

    async def _decide_and_execute(self, context: dict) -> None:
        letters: list[Letter] = context["letters"]
        draft_paths: list[Path] = context["drafts"]

        if not letters and not draft_paths:
            return

        # Read draft contents
        drafts: list[tuple[Path, str]] = []
        for p in draft_paths:
            try:
                drafts.append((p, p.read_text(encoding="utf-8")))
            except OSError:
                continue

        # Build user prompt — correspondence as the character would experience it
        prompt_parts: list[str] = []

        if letters:
            prompt_parts.append("You have letters:")
            for letter in letters:
                # Extract sender name from filename convention
                sender = self._parse_sender(letter.filename)
                body = letter.body.strip()
                prompt_parts.append(f"\nFrom {sender}:\n{body}")

        if drafts:
            prompt_parts.append("\nYou have unsent letters you wrote earlier:")
            for path, content in drafts:
                # Parse recipient from draft content
                recipient = self._parse_draft_recipient(content)
                body = self._parse_draft_body(content)
                prompt_parts.append(f"\nTo {recipient}:\n{body}")

        prompt_parts.append(
            "\nFor each letter: decide whether to reply.\n"
            "For each unsent letter: decide whether to send it, wait, or let it go.\n\n"
            "To reply: [REPLY TO: sender name | your reply]\n"
            "To send: [SEND: recipient name]\n"
            "To hold: [HOLD: recipient name]\n"
            "To discard: [DISCARD: recipient name]"
        )

        # System prompt: just the personality paragraph — no full soul needed for correspondence
        system_prompt = self._extract_personality(self._identity.soul)
        user_prompt = "\n".join(prompt_parts)

        response = await self._llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self._tuning.mail_model,
            temperature=self._tuning.mail_temperature,
            max_tokens=self._tuning.mail_max_tokens,
        )

        await self._process_mail_response(response, letters, drafts)

    async def _process_mail_response(
        self,
        response: str,
        letters: list[Letter],
        drafts: list[tuple[Path, str]],
    ) -> None:
        # Send replies
        for match in _RE_REPLY.finditer(response):
            sender_name = match.group(1).strip()
            reply_body = match.group(2).strip()

            # Find the session ID to reply to from the letter body
            to_session = self._find_reply_session(sender_name, letters)
            if to_session:
                try:
                    await self._ww.reply_letter(self.name, to_session, reply_body)
                    logger.info("[%s:mail] replied to %s", self.name, sender_name)
                except Exception as e:
                    logger.warning("[%s:mail] reply to %s failed: %s", self.name, sender_name, e)
            else:
                logger.debug("[%s:mail] no reply session found for %s", self.name, sender_name)

        # Process draft decisions
        for path, content in drafts:
            recipient = self._parse_draft_recipient(content)

            if _RE_SEND.search(response) and recipient.lower() in response.lower():
                await self._send_draft(path, content)
            elif _RE_DISCARD.search(response) and recipient.lower() in response.lower():
                path.unlink(missing_ok=True)
                logger.info("[%s:mail] discarded draft to %s", self.name, recipient)
            elif _RE_HOLD.search(response) and recipient.lower() in response.lower():
                logger.info("[%s:mail] holding draft to %s", self.name, recipient)
            # No match = implicit hold — leave it

    async def _send_draft(self, path: Path, content: str) -> None:
        recipient = self._parse_draft_recipient(content)
        body = self._parse_draft_body(content)

        try:
            await self._ww.send_letter(
                from_name=self.name,
                to_agent=recipient,
                body=body,
                session_id=self._session_id,
            )
            self._sent_dir.mkdir(parents=True, exist_ok=True)
            path.rename(self._sent_dir / path.name)
            logger.info("[%s:mail] sent letter to %s", self.name, recipient)
        except Exception as e:
            logger.warning("[%s:mail] send to %s failed: %s", self.name, recipient, e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_sender(self, filename: str) -> str:
        """Extract sender name from letter filename convention: from_{name}_{ts}.md"""
        stem = Path(filename).stem  # e.g. from_margot_20260309-121500
        if stem.startswith("from_"):
            parts = stem[5:].split("_")
            return parts[0].capitalize() if parts else filename
        return filename

    def _parse_draft_recipient(self, content: str) -> str:
        for line in content.splitlines():
            if line.startswith("To:"):
                return line[3:].strip()
        return "unknown"

    def _parse_draft_body(self, content: str) -> str:
        lines = content.splitlines()
        # Skip header lines (To:, Staged-At:, blank line)
        in_body = False
        body_lines = []
        for line in lines:
            if in_body:
                body_lines.append(line)
            elif line.strip() == "":
                in_body = True
        return "\n".join(body_lines).strip()

    def _find_reply_session(self, sender_name: str, letters: list[Letter]) -> str | None:
        """Look for Reply-To-Session header in the letter from this sender."""
        for letter in letters:
            if sender_name.lower() in letter.filename.lower():
                for line in letter.body.splitlines():
                    if line.startswith("Reply-To-Session:"):
                        return line.split(":", 1)[1].strip()
        return None

    def _extract_personality(self, soul: str) -> str:
        """
        Use just the personality paragraph for the mail loop system prompt.
        The full soul is more than needed for correspondence triage.
        Falls back to the full soul if no paragraph structure is found.
        """
        # Look for the ## Personality section
        match = re.search(r'## Personality\s+(.+?)(?=\n##|\Z)', soul, re.DOTALL)
        if match:
            return match.group(1).strip()
        return soul

    async def _cooldown(self) -> None:
        await asyncio.sleep(self._tuning.mail_send_delay_seconds)
