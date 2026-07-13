"""Semantic search with the [semantic] extra installed (SPEC §5.4)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager

from agentplane_registry.app import create_app
from agentplane_registry.search import semantic_available

from .conftest import agent_entry_body, make_settings

EMBEDDINGS_URL = "http://gateway.test/v1"

pytestmark = pytest.mark.skipif(not semantic_available(), reason="[semantic] extra not installed")

# Deterministic fake embeddings: direction encodes the topic.
_VECTORS = {
    "billing": [1.0, 0.0, 0.0],
    "invoices": [0.9, 0.1, 0.0],
    "weather": [0.0, 1.0, 0.0],
}


def _vector_for(text: str) -> list[float]:
    lowered = text.lower()
    for keyword, vector in _VECTORS.items():
        if keyword in lowered:
            return vector
    return [0.0, 0.0, 1.0]


def _mock_embeddings() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        vector = _vector_for(str(payload["input"]))
        return httpx.Response(200, json={"data": [{"embedding": vector}]})

    respx.post(f"{EMBEDDINGS_URL}/embeddings").mock(side_effect=responder)


@pytest.fixture
async def semantic_client() -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(
        embeddings_base_url=EMBEDDINGS_URL, embeddings_model="text-embedding-3-small"
    )
    app = create_app(settings, run_health_job=False)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://registry.test"
        ) as client:
            yield client


@respx.mock
async def test_capabilities_report_semantic(semantic_client: httpx.AsyncClient) -> None:
    caps = (await semantic_client.get("/api/v1/capabilities")).json()
    assert caps["semantic_search"] is True


@respx.mock
async def test_semantic_search_ranks_by_similarity(semantic_client: httpx.AsyncClient) -> None:
    _mock_embeddings()
    billing = agent_entry_body("billing-agent")
    billing["card"]["description"] = "Handles billing questions"
    weather = agent_entry_body("weather-agent")
    weather["card"]["description"] = "Reports the weather"
    assert (await semantic_client.post("/api/v1/agents", json=billing)).status_code == 201
    assert (await semantic_client.post("/api/v1/agents", json=weather)).status_code == 201

    response = await semantic_client.get(
        "/api/v1/agents/search", params={"q": "invoices", "semantic": "true"}
    )
    assert response.status_code == 200
    assert "X-Degraded" not in response.headers
    hits = response.json()
    # "invoices" has no text match at all — only the semantic ranking finds it
    assert hits["total"] >= 1
    assert hits["items"][0]["card"]["name"] == "billing-agent"
