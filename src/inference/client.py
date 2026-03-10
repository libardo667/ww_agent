from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    pass


class InferenceClient:
    """
    Thin async wrapper around an OpenRouter-compatible chat completions API.
    Shared by all residents and all loops. Stateless — context comes from caller.

    The agent never knows this exists. All prompt construction happens in the
    loop layer; this client just sends text and returns text.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str = "google/gemini-flash-1.5",
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ) -> str:
        """
        Send a chat completion. Returns the assistant message text.
        Retries on transient errors (429, 500, 502, 503).
        """
        payload = {
            "model": model or self._default_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = await self._post_with_retry("/chat/completions", payload)
        content = response["choices"][0]["message"]["content"]

        usage = response.get("usage", {})
        logger.debug(
            "inference: model=%s tokens=%s+%s",
            payload["model"],
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )

        return content

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> dict:
        """
        Like complete(), but parses the response as JSON.
        Strips markdown fences if present. Raises InferenceError on parse failure.

        Use sparingly — "respond with JSON" is the most visible seam.
        Prefer complete() + lightweight parsing where possible.
        """
        text = await self.complete(system_prompt, user_prompt, **kwargs)
        text = text.strip()

        # Strip markdown fences if the model wrapped the JSON
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise InferenceError(f"Response was not valid JSON: {e}\n\nResponse was:\n{text}") from e

    async def _post_with_retry(
        self,
        path: str,
        payload: dict,
        *,
        max_retries: int = 2,
    ) -> dict:
        retryable = {429, 500, 502, 503}
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.post(path, json=payload)

                if resp.status_code in retryable:
                    delay = 2 ** attempt
                    logger.warning(
                        "inference: HTTP %s, retrying in %ss (attempt %s/%s)",
                        resp.status_code, delay, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(delay)
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise InferenceError("Inference request timed out") from e

        raise InferenceError(f"Inference failed after {max_retries + 1} attempts") from last_error

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> InferenceClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
