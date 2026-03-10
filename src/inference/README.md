# src/inference/ — LLM Inference Client

## Purpose

Thin async wrapper around an OpenRouter-compatible chat completions API.
Shared by all residents and all loops. Stateless — all context comes from
the caller.

## Interface Contract

```python
class InferenceClient:
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,       # override default model
        temperature: float = 0.7,
        max_tokens: int = 300,
        response_format: str = "text",   # "text" or "json"
    ) -> str:
        """Send a chat completion request. Returns the assistant message content."""

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> dict:
        """Like complete(), but parses the response as JSON.
        Strips markdown fences if present. Raises on parse failure."""
```

## Implementation Notes

- Use `httpx.AsyncClient` with a shared connection pool.
- Set reasonable timeouts (30s for fast loop, 60s for slow loop).
- Retry on 429/500/502/503 with exponential backoff (max 2 retries).
- Log token usage per call for cost tracking.
- The client does NOT know about loops, residents, or WorldWeaver.
  It's a pure HTTP wrapper.

## Model Selection

Default model comes from config. Each loop can override via tuning.json:

```json
{
    "fast": { "model": "google/gemini-3-flash-preview", "temperature": 0.8 },
    "slow": { "model": "google/gemini-3-flash-preview", "temperature": 0.6 },
    "mail": { "model": "google/gemini-3-flash-preview", "temperature": 0.5 }
}
```

In practice, Gemini Flash for everything is fine. The temperature differences
matter more than model differences for personality expression.

## On `complete` vs `complete_json`

`complete_json` is the visible seam. Telling the agent "respond with a JSON
object" is the most direct way to remind them they're producing structured
output for a machine. Use it sparingly and only where the output is genuinely
list-like (mail loop triage decisions are a reasonable case).

For fast loop output: use `complete()` and parse the plain text yourself.
A short prose response — "She turns toward the sound." — doesn't need JSON.

For slow loop output: consider lightweight conventions that feel literary
rather than mechanical. A bracketed tag at the end of a reflective paragraph
(`[ACTION: I walk toward the iron door]`) is less jarring than asking the
agent to produce a full object. Structured extraction can happen in the
framework after the response arrives, not in the agent's awareness before
they write it.

The test: if you removed the output format instruction from the prompt, would
the agent naturally produce something parseable anyway? If yes, remove the
instruction. Parse what you get. That's the naturalized path.
