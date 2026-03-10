# src/ — Agent Runtime Source

## Module Map

```
src/
├── main.py              Entry point. Loads residents, starts loops, runs forever.
├── resident.py          Resident lifecycle: load identity, init loops, manage state.
├── loops/
│   ├── README.md        You're here. Loop architecture and contracts.
│   ├── base.py          Abstract base loop with trigger/execute/cooldown pattern.
│   ├── fast.py          Fast loop: scene-local reaction, reflex, tiny context.
│   ├── slow.py          Slow loop: reflection, goal-setting, identity evolution.
│   └── mail.py          Mail loop: inbox triage, draft send/discard, correspondence.
├── inference/
│   ├── README.md        LLM client docs.
│   └── client.py        Thin async wrapper around OpenRouter/compatible API.
├── world/
│   ├── README.md        WorldWeaver client docs.
│   └── client.py        Async HTTP client for WorldWeaver /api/* endpoints.
├── memory/
│   ├── README.md        Memory system docs.
│   ├── working.py       Rolling buffer of recent events (small, fast).
│   ├── provisional.py   Fast-loop scratchpad: write impressions, read/archive them.
│   └── retrieval.py     Semantic search over long-term memory (slow loop only).
└── identity/
    ├── README.md        Identity system docs.
    └── loader.py        Load SOUL.md, IDENTITY.md, tuning.json at startup.
```

## Concurrency Model

All loops run as asyncio tasks within a single process. No threads, no
subprocesses, no inter-process communication. Loops communicate only through
the filesystem (provisional impressions, letter drafts) — never through
shared memory or message passing.

## Entry Point Contract

`main.py` must:

1. Parse CLI args / env vars for server URL, inference config, residents dir.
2. Discover all resident directories in the residents dir.
3. For each resident: load identity, create loop instances, register with server.
4. Start all loops as asyncio tasks.
5. Run forever (or until SIGTERM/SIGINT).
6. On shutdown: gracefully cancel all loops, flush any pending writes.

```python
# Pseudocode for main.py
async def main():
    config = load_config()
    ww_client = WorldWeaverClient(config.server_url)
    llm_client = InferenceClient(config.inference_url, config.inference_key)

    residents = discover_residents(config.residents_dir)
    tasks = []
    for resident in residents:
        r = Resident.load(resident.path, ww_client, llm_client)
        tasks.extend(r.start_loops())

    await asyncio.gather(*tasks)
```
