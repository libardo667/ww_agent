from __future__ import annotations

import json
import random
from pathlib import Path


class ReverieDeck:
    """
    A rotating deck of personal sensory/emotional moments accumulated by
    the slow loop. The fast loop draws from it as a live reverie anchor
    instead of the static identity.core prose.

    Each reverie is a short first-person image: something the character
    noticed, felt, or carried away from an experience. They vary per agent
    and evolve with the character's history — the opposite of a fixed
    identity paragraph.

    Falls back to None when empty (fast loop uses identity.core instead).
    """

    MAX_REVERIES = 20

    def __init__(self, path: Path) -> None:
        self._path = path

    def add(self, text: str) -> None:
        """Append a new reverie, trimming the deck to MAX_REVERIES."""
        reveries = self._load()
        reveries.append(text.strip())
        if len(reveries) > self.MAX_REVERIES:
            reveries = reveries[-self.MAX_REVERIES:]
        self._save(reveries)

    def random_pick(self) -> str | None:
        """Return a random reverie, or None if the deck is empty."""
        reveries = self._load()
        return random.choice(reveries) if reveries else None

    def __len__(self) -> int:
        return len(self._load())

    def _load(self) -> list[str]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, reveries: list[str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(reveries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
