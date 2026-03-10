"""
ww_agent — WorldWeaver resident daemon

Boots one or more residents and (optionally) the doula loop.
Configuration is via environment variables.

Required:
    WW_URL          WorldWeaver server base URL (e.g. http://localhost:8000)
    LLM_URL         OpenRouter-compatible API base URL
    LLM_KEY         API key for the LLM provider

Optional:
    LLM_MODEL       Default model (default: google/gemini-flash-1.5)
    RESIDENTS_DIR   Path to directory containing resident subdirectories
                    (default: ./residents)
    DOULA           Enable the doula loop ("1" / "true" to enable)
    DOULA_MODEL     Model override for soul seeding
    LOG_LEVEL       Logging level (default: INFO)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from src.inference.client import InferenceClient
from src.loops.doula import DoulaLoop
from src.resident import Resident
from src.world.client import WorldWeaverClient


def _configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _discover_residents(residents_dir: Path) -> list[Path]:
    """
    Return subdirectories that look like resident homes.
    A valid resident dir has identity/SOUL.md.
    """
    found = []
    for entry in sorted(residents_dir.iterdir()):
        if entry.is_dir() and (entry / "identity" / "SOUL.md").exists():
            found.append(entry)
    return found


async def _boot_resident(
    resident_dir: Path,
    ww_client: WorldWeaverClient,
    llm: InferenceClient,
    world_id: str,
) -> asyncio.Task:
    """Start a resident and return the running task."""
    r = Resident(resident_dir, ww_client, llm)
    await r.start(world_id)
    task = asyncio.create_task(r.run(), name=f"resident:{r.name}")
    return task


async def _drain_spawn_queue(
    spawn_queue: asyncio.Queue,
    ww_client: WorldWeaverClient,
    llm: InferenceClient,
    world_id: str,
    running_tasks: set[asyncio.Task],
) -> None:
    """Continuously accept new residents from the doula's spawn queue."""
    while True:
        resident_dir: Path = await spawn_queue.get()
        logging.getLogger(__name__).info("doula spawn: booting %s", resident_dir.name)
        try:
            task = await _boot_resident(resident_dir, ww_client, llm, world_id)
            running_tasks.add(task)
            task.add_done_callback(running_tasks.discard)
        except Exception as e:
            logging.getLogger(__name__).warning("failed to boot %s: %s", resident_dir.name, e)


async def main() -> None:
    _configure_logging()
    log = logging.getLogger(__name__)

    # -- Config --
    ww_url       = os.environ.get("WW_URL", "http://localhost:8000")
    llm_url      = os.environ.get("LLM_URL", "https://openrouter.ai/api/v1")
    llm_key      = os.environ.get("LLM_KEY", "")
    llm_model    = os.environ.get("LLM_MODEL", "google/gemini-flash-1.5")
    residents_dir = Path(os.environ.get("RESIDENTS_DIR", "residents"))
    doula_enabled = os.environ.get("DOULA", "").lower() in ("1", "true", "yes")
    doula_model  = os.environ.get("DOULA_MODEL") or None

    if not llm_key:
        log.error("LLM_KEY is required")
        sys.exit(1)

    if not residents_dir.exists():
        log.error("RESIDENTS_DIR does not exist: %s", residents_dir)
        sys.exit(1)

    # -- Shared clients --
    ww_client = WorldWeaverClient(base_url=ww_url)
    llm = InferenceClient(base_url=llm_url, api_key=llm_key, default_model=llm_model)

    log.info("waiting for WorldWeaver at %s", ww_url)
    await ww_client.wait_for_ready(timeout_seconds=60.0)

    world_id = await ww_client.get_world_id()
    if not world_id:
        log.error("No world ID found — is a world seeded on the server?")
        sys.exit(1)

    log.info("world: %s", world_id)

    # -- Boot existing residents --
    resident_dirs = _discover_residents(residents_dir)
    if not resident_dirs:
        log.warning("no residents found in %s", residents_dir)

    running_tasks: set[asyncio.Task] = set()

    tethered_names: set[str] = set()
    session_ids: list[str] = []

    for resident_dir in resident_dirs:
        try:
            task = await _boot_resident(resident_dir, ww_client, llm, world_id)
            running_tasks.add(task)
            task.add_done_callback(running_tasks.discard)
            tethered_names.add(resident_dir.name)

            # Collect session IDs for doula proximity checks
            session_path = resident_dir / "session_id.txt"
            if session_path.exists():
                session_ids.append(session_path.read_text(encoding="utf-8").strip())
        except Exception as e:
            log.warning("failed to boot resident %s: %s", resident_dir.name, e)

    log.info("booted %d residents", len(running_tasks))

    # -- Doula loop --
    spawn_queue: asyncio.Queue = asyncio.Queue()
    all_tasks = list(running_tasks)

    if doula_enabled:
        doula = DoulaLoop(
            ww_client=ww_client,
            llm=llm,
            residents_dir=residents_dir,
            spawn_queue=spawn_queue,
            tethered_names=tethered_names,
            known_session_ids=session_ids,
            soul_model=doula_model,
        )
        doula_task = asyncio.create_task(doula.run(), name="doula")
        spawn_drain = asyncio.create_task(
            _drain_spawn_queue(spawn_queue, ww_client, llm, world_id, running_tasks),
            name="doula:spawn-drain",
        )
        all_tasks += [doula_task, spawn_drain]
        log.info("doula loop enabled")

    # -- Run until interrupted --
    try:
        await asyncio.gather(*all_tasks)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        log.info("shutting down")
        for task in running_tasks:
            task.cancel()
        await ww_client.close()
        await llm.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
