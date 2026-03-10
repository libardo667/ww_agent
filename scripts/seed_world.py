"""
seed_world.py — Seed a normcore WorldWeaver world and optionally reset residents.

Calls:
    POST /api/dev/hard-reset   (optional — wipes all world data)
    POST /api/world/seed       (creates world_id + storylets in one shot)

Then optionally resets all resident runtime state (session_id.txt, memory,
letters, decisions) so they start fresh in the new world.

Usage:
    python scripts/seed_world.py [OPTIONS]

    --server URL       WorldWeaver server URL (default: http://localhost:8000)
    --no-reset         Skip the hard-reset (add to an existing world instead)
    --no-residents     Skip resetting resident runtime state
    --residents-dir D  Path to residents directory (default: ./residents)
    --theme TEXT       World theme override
    --tone TEXT        World tone override
    --count N          Number of storylets to generate (default: 20)
    --dry-run          Print payload without calling the server
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Defaults — edit these to change the world without touching the CLI flags
# ---------------------------------------------------------------------------

DEFAULT_SERVER = "http://localhost:8000"

DEFAULT_THEME = "Everyday life in San Francisco's Mission District, grounded in real places — Dolores Park, taquerias, the BART, corner laundromats, weekend farmers markets."

DEFAULT_PLAYER_ROLE = "A resident of the neighborhood, living an ordinary life."

DEFAULT_TONE = "quiet and observational; everyday life without manufactured drama"

DEFAULT_DESCRIPTION = (
    "A persistent, shared neighborhood where people live their lives. "
    "Characters walk to the park, grab coffee, run errands, sit on stoops. "
    "The world accumulates a quiet history through these small acts. "
    "Do not invent conflict or drama — let the texture of ordinary life be enough."
)

DEFAULT_STORYLET_COUNT = 20


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no httpx dependency here)
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _post_empty(url: str) -> dict:
    req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Resident runtime reset
# ---------------------------------------------------------------------------

_RUNTIME_DIRS = ("memory", "letters", "decisions", "turns")
_RUNTIME_FILES = ("session_id.txt", "world_id.txt")


def _restore_soul(resident_dir: Path, dry_run: bool) -> None:
    """Truncate SOUL.md to canonical content (everything before the first '---' line)."""
    soul_path = resident_dir / "identity" / "SOUL.md"
    if not soul_path.exists():
        return
    text = soul_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    canonical: list[str] = []
    for line in lines:
        if line.rstrip() == "---":
            break
        canonical.append(line)
    restored = "".join(canonical).rstrip("\n") + "\n"
    if restored == text:
        return  # nothing to strip
    print(f"  soul restore: {soul_path.relative_to(resident_dir.parent.parent)}")
    if not dry_run:
        soul_path.write_text(restored, encoding="utf-8")


def _reset_resident(resident_dir: Path, dry_run: bool) -> None:
    name = resident_dir.name
    for d in _RUNTIME_DIRS:
        target = resident_dir / d
        if target.exists():
            print(f"  rm -rf {target.relative_to(resident_dir.parent.parent)}")
            if not dry_run:
                shutil.rmtree(target)
    for f in _RUNTIME_FILES:
        target = resident_dir / f
        if target.exists():
            print(f"  rm {target.relative_to(resident_dir.parent.parent)}")
            if not dry_run:
                target.unlink()
    _restore_soul(resident_dir, dry_run)
    print(f"  [ok] {name} reset")


def _reset_all_residents(residents_dir: Path, dry_run: bool) -> None:
    found = [
        d for d in residents_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "identity" / "SOUL.md").exists()
    ]
    if not found:
        print("No residents found to reset.")
        return
    print(f"\nResetting {len(found)} resident(s):")
    for resident_dir in sorted(found):
        _reset_resident(resident_dir, dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a normcore WorldWeaver world.")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--no-reset", action="store_true", help="Skip hard-reset (keep existing world data)")
    parser.add_argument("--no-residents", action="store_true", help="Skip resetting resident runtime state")
    parser.add_argument("--residents-dir", default="./residents")
    parser.add_argument("--theme", default=DEFAULT_THEME)
    parser.add_argument("--tone", default=DEFAULT_TONE)
    parser.add_argument("--count", type=int, default=DEFAULT_STORYLET_COUNT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    server = args.server.rstrip("/")

    # 1. Hard reset
    if not args.no_reset:
        print(f"[1/3] Hard reset: POST {server}/api/dev/hard-reset")
        if not args.dry_run:
            try:
                result = _post_empty(f"{server}/api/dev/hard-reset")
                print(f"      ok: {result}")
            except urllib.error.HTTPError as e:
                body = e.read().decode()
                print(f"      ERROR {e.code}: {body}", file=sys.stderr)
                sys.exit(1)
        else:
            print("      [dry-run skipped]")
    else:
        print("[1/3] Skipping hard-reset (--no-reset)")

    # 2. Seed world
    seed_payload = {
        "world_theme": args.theme,
        "player_role": DEFAULT_PLAYER_ROLE,
        "description": DEFAULT_DESCRIPTION,
        "tone": args.tone,
        "storylet_count": args.count,
    }
    print(f"\n[2/3] Seed world: POST {server}/api/world/seed")
    print("      payload:")
    for k, v in seed_payload.items():
        short = v if len(str(v)) <= 80 else str(v)[:77] + "..."
        print(f"        {k}: {short}")

    world_id = None
    if not args.dry_run:
        try:
            result = _post(f"{server}/api/world/seed", seed_payload)
            world_id = result.get("world_id")
            storylet_count = result.get("storylet_count", "?")
            print(f"      ok: world_id={world_id}  storylets={storylet_count}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"      ERROR {e.code}: {body}", file=sys.stderr)
            sys.exit(1)
    else:
        print("      [dry-run skipped]")

    # 3. Reset residents
    residents_dir = Path(args.residents_dir)
    if not args.no_residents:
        if residents_dir.exists():
            _reset_all_residents(residents_dir, args.dry_run)
        else:
            print(f"\n[3/3] Residents dir not found: {residents_dir} — skipping")
    else:
        print("\n[3/3] Skipping resident reset (--no-residents)")

    print(f"\nDone. World is ready.")
    if world_id:
        print(f"  world_id: {world_id}")
    print("  Start ww_agent to boot residents into the new world.")


if __name__ == "__main__":
    main()
