# tests/ — Test Suite

## Test Strategy

### Unit Tests (no network, no LLM)

- `test_working_memory.py` — buffer append, eviction, persistence, recent()
- `test_provisional.py` — write impression, read pending, promote/archive/discard
- `test_identity_loader.py` — load SOUL.md, IDENTITY.md, tuning.json, handle missing files
- `test_tuning.py` — parse tuning.json, defaults for missing fields, personality presets
- `test_loop_contracts.py` — verify capability enforcement (fast loop cannot access mail client, etc.)

### Integration Tests (mock LLM, mock WorldWeaver)

- `test_fast_loop.py` — feed a scene, verify single short action, verify provisional written
- `test_slow_loop.py` — seed provisional impressions, verify processing (promote/archive/discard)
- `test_mail_loop.py` — stage drafts, verify send/discard decisions, verify inbox processing
- `test_resident_lifecycle.py` — boot a resident, run N cycles, verify file artifacts

### Smoke Tests (real server, real LLM)

- `test_smoke_single_resident.py` — boot one resident against a running WorldWeaver, play 5 turns
- `test_smoke_multi_resident.py` — boot 3 residents, verify they colocate and react to each other

## Running Tests

```bash
# Unit + integration (no external dependencies)
pytest tests/ -m "not smoke"

# Smoke tests (requires running WorldWeaver server + API key)
WW_INFERENCE_KEY=sk-or-... pytest tests/ -m smoke
```

## Key Invariants to Test

1. Fast loop NEVER writes to SOUL.md (capability contract)
2. Mail loop NEVER posts to /api/action (capability contract)
3. Working memory never exceeds max_items
4. Provisional impressions transition: pending → promoted/archived/discarded (never stuck)
5. Slow loop processes ALL pending impressions each cycle (no leaks)
6. Resident with mail.enabled=false never instantiates a mail loop
7. Multiple residents in same process don't share memory state
8. SOUL.md mutations by slow loop are persisted and visible on next load
