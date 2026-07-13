"""Entry embeddings via an OpenAI-compatible /embeddings endpoint (SPEC §5.4)."""

from __future__ import annotations

import httpx

from agentplane_core import RegistryEntry, ToolCard


def embedding_text(entry: RegistryEntry) -> str:
    """``f"{name}\\n{description}\\n" + skill descriptions`` (SPEC §5.4)."""
    card = entry.card
    if isinstance(card, ToolCard):
        skills = [f"{tool.name} {tool.description}".strip() for tool in card.tools]
        name, description = card.name, card.description
    else:
        skills = [f"{skill.name} {skill.description}".strip() for skill in card.skills]
        name, description = card.name, card.description
    return f"{name}\n{description}\n" + "\n".join(skills)


class EmbeddingsClient:
    """Minimal OpenAI-compatible embeddings client, pointed at the gateway."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                json={"model": self.model, "input": text},
            )
            response.raise_for_status()
        data = response.json()["data"]
        vector = data[0]["embedding"]
        return [float(v) for v in vector]


__all__ = ["EmbeddingsClient", "embedding_text"]
