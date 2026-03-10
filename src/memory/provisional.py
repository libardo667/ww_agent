from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Impression:
    path: Path
    ts: str
    trigger: str        # what the fast loop noticed
    raw_reaction: str   # its immediate, uninterpreted response
    location: str
    colocated: list[str]

    @classmethod
    def from_file(cls, path: Path) -> Impression:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            ts=data["ts"],
            trigger=data["trigger"],
            raw_reaction=data["raw_reaction"],
            location=data["location"],
            colocated=data.get("colocated", []),
        )

    def as_prose(self) -> str:
        """Naturalized form for slow loop context — no system vocabulary."""
        others = ", ".join(self.colocated) if self.colocated else "no one else"
        return f"{self.raw_reaction} (in {self.location}, with {others})"


class ProvisionalScratchpad:
    """
    Fast loop writes raw impressions here. Slow loop reads and processes them.
    The agent never sees file paths or status fields — only the prose content.
    """

    def __init__(self, provisional_dir: Path):
        self._dir = provisional_dir
        self._archive_dir = provisional_dir / "archived"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

    def write_impression(
        self,
        trigger: str,
        raw_reaction: str,
        location: str,
        colocated: list[str],
    ) -> Path:
        """Fast loop: record a raw, uninterpreted moment."""
        ts = datetime.now(timezone.utc).isoformat().replace(":", "-").replace("+", "Z").split("Z")[0] + "Z"
        filename = f"imp_{ts}.json"
        path = self._dir / filename
        data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
            "raw_reaction": raw_reaction,
            "location": location,
            "colocated": colocated,
            "status": "pending",
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def pending_impressions(self) -> list[Impression]:
        """Slow loop: read all unprocessed impressions."""
        impressions = []
        for p in sorted(self._dir.glob("imp_*.json")):
            try:
                imp = Impression.from_file(p)
                impressions.append(imp)
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return impressions

    def promote(self, impression: Impression, decision_note: str) -> None:
        """Slow loop: this impression became a decision. Remove from pending."""
        # The decision log is written separately by the slow loop.
        # We just clean up the pending file.
        impression.path.unlink(missing_ok=True)

    def archive(self, impression: Impression, interpretation: str) -> None:
        """Slow loop: worth keeping but didn't drive a decision."""
        data = json.loads(impression.path.read_text(encoding="utf-8"))
        data["status"] = "archived"
        data["interpretation"] = interpretation
        dest = self._archive_dir / impression.path.name
        dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        impression.path.unlink(missing_ok=True)

    def discard(self, impression: Impression) -> None:
        """Slow loop: not worth keeping."""
        impression.path.unlink(missing_ok=True)

    def pending_as_prose(self) -> str:
        """
        Naturalized context string for the slow loop prompt.
        Returns impressions as flowing text the character would recognize
        as their own recent experience — no file paths, no status fields.
        """
        pending = self.pending_impressions()
        if not pending:
            return ""
        lines = [imp.as_prose() for imp in pending]
        return "Lately you've been noticing:\n" + "\n".join(f"- {l}" for l in lines)
