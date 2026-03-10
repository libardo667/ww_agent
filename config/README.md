# config/ — Runtime Configuration

## Philosophy

No config files. All runtime configuration comes from environment variables
or CLI flags. This directory exists only for documentation and optional
env file templates.

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `WW_INFERENCE_KEY` | API key for LLM provider (OpenRouter, etc.) |

### Optional (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `WW_SERVER_URL` | `http://localhost:8000/api` | WorldWeaver API base URL |
| `WW_INFERENCE_URL` | `https://openrouter.ai/api/v1` | LLM API base URL |
| `WW_INFERENCE_MODEL` | `google/gemini-3-flash-preview` | Default model |
| `WW_RESIDENTS_DIR` | `./residents` | Path to residents directory |
| `WW_LOG_LEVEL` | `info` | Logging: debug, info, warning, error |
| `WW_WORLD_ID` | (auto-detected) | Override shared world ID |

## Template .env

Copy `env.example` and fill in your values:

```bash
cp config/env.example .env
```

The daemon reads `.env` from the working directory if present (via python-dotenv).
Environment variables always take precedence over the file.

## Why No Config File

Config files create state. State creates drift. Drift creates debugging.
The daemon should be fully defined by its inputs (env vars + resident
directories) and produce fully predictable behavior. If you can't reproduce
a run by setting the same env vars and pointing at the same residents
directory, something is wrong.
