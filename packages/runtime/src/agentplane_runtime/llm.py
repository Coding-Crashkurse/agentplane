"""OpenAI-compatible LLM access — always through a configured base URL (the gateway)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from agentplane_core import JsonSchema


class LlmError(RuntimeError):
    """Chat completion or embedding request failed."""


class OpenAICompatibleClient:
    """Minimal chat-completions + embeddings client (httpx, async, SSE streaming)."""

    def __init__(self, base_url: str, api_key: str = "", *, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _chat_body(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        structured_output: JsonSchema | None,
        *,
        stream: bool,
    ) -> dict[str, object]:
        messages: list[dict[str, object]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, object] = {"model": model, "messages": messages, "stream": stream}
        if structured_output is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "schema": structured_output},
            }
        return body

    async def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        structured_output: JsonSchema | None = None,
    ) -> str:
        body = self._chat_body(model, prompt, system_prompt, structured_output, stream=False)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions", json=body, headers=self._headers()
            )
        if response.status_code != httpx.codes.OK:
            raise LlmError(f"chat completion failed: HTTP {response.status_code}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LlmError(f"malformed chat completion response: {exc}") from exc
        return str(content or "")

    async def stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        structured_output: JsonSchema | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas from an SSE chat-completions stream."""
        body = self._chat_body(model, prompt, system_prompt, structured_output, stream=True)
        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=body,
                headers=self._headers(),
            ) as response,
        ):
            if response.status_code != httpx.codes.OK:
                raise LlmError(f"chat completion stream failed: HTTP {response.status_code}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, ValueError):
                    continue
                if delta:
                    yield str(delta)

    async def embed(self, model: str, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                json={"model": model, "input": text},
                headers=self._headers(),
            )
        if response.status_code != httpx.codes.OK:
            raise LlmError(f"embedding failed: HTTP {response.status_code}")
        try:
            vector = response.json()["data"][0]["embedding"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LlmError(f"malformed embeddings response: {exc}") from exc
        return [float(v) for v in vector]


__all__ = ["LlmError", "OpenAICompatibleClient"]
