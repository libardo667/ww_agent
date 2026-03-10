# scripts/ — Utility Scripts

## Planned Scripts

### spawn_resident.sh

Create a new resident from the template:

```bash
./scripts/spawn_resident.sh nadia
# Creates residents/nadia/ from residents/_template/
# Opens SOUL.md in $EDITOR for immediate editing
```

### export_transcript.py

Export a resident's turn history as a readable markdown transcript:

```bash
python scripts/export_transcript.py --resident nadia --output transcripts/nadia.md
```

### export_letters.py

Collect all letters written by a resident into a single document:

```bash
python scripts/export_letters.py --resident nadia --output letters/nadia_collected.md
```

### world_status.py

Query the WorldWeaver server and print a summary of active sessions,
locations, and recent events:

```bash
python scripts/world_status.py --server http://localhost:8000/api
```

### reset_resident.py

Wipe a resident's memory and runtime state while preserving their identity
(SOUL.md, IDENTITY.md, tuning.json). This is a "reincarnation" — same
person, new life, no memories:

```bash
python scripts/reset_resident.py --resident nadia
# Deletes: memory/, letters/, decisions/, turns/, session_id.txt, world_id.txt
# Preserves: identity/
```

### bulk_reset.py

Reset all residents for a new world:

```bash
python scripts/bulk_reset.py --residents-dir ./residents
# Same as reset_resident for every directory in residents/ except _template/
```
