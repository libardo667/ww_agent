from __future__ import annotations

import json
import random
from pathlib import Path


class VoiceDeck:
    """
    A living deck of speech-sample utterances for one resident.

    Mirrors ReverieDeck but for the mouth rather than the inner eye.
    Each entry is a short, actual sentence the character would say aloud —
    seeded from IDENTITY.md on first boot and evolved by the slow loop
    extracting real utterances from chat history.

    The fast loop draws a handful of samples and injects them into the
    system prompt when chat context is present, giving the LLM concrete
    examples of register rather than abstract soul prose.

    Falls back to [] when empty (fast loop omits the voice block).
    """

    MAX_SAMPLES = 30

    def __init__(self, path: Path) -> None:
        self._path = path

    def add(self, text: str) -> None:
        """Append a new sample, trimming to MAX_SAMPLES."""
        samples = self._load()
        text = text.strip().strip('"\'')
        if not text or text in samples:
            return
        samples.append(text)
        if len(samples) > self.MAX_SAMPLES:
            samples = samples[-self.MAX_SAMPLES:]
        self._save(samples)

    def seed(self, utterances: list[str]) -> None:
        """Seed from IDENTITY.md voice field — only if deck is currently empty."""
        if self._load():
            return
        for u in utterances:
            self.add(u)

    def sample(self, n: int = 3) -> list[str]:
        """Return up to n random samples, or [] if deck is empty."""
        samples = self._load()
        if not samples:
            return []
        k = min(n, len(samples))
        return random.sample(samples, k)

    def __len__(self) -> int:
        return len(self._load())

    def _load(self) -> list[str]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, samples: list[str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
        )
