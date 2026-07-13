"""Resources API (SPEC §6.3): CRUD, write-only secrets, delete protection."""

from __future__ import annotations

import httpx

from .conftest import create_default_resources, llm_resource_body, load_example, vector_db_body


async def test_create_returns_redacted_secret(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/resources", json=llm_resource_body())
    assert response.status_code == 201
    body = response.json()
    assert body["api_key_secret"] == "•••"
    assert "sk-secret" not in response.text


async def test_get_and_list_stay_redacted(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    single = (await client.get("/api/v1/resources/default-llm")).json()
    assert single["api_key_secret"] == "•••"
    listing = (await client.get("/api/v1/resources")).json()
    assert {r["name"] for r in listing} == {"default-llm", "kb-support"}
    filtered = (await client.get("/api/v1/resources", params={"kind": "qdrant"})).json()
    assert [r["name"] for r in filtered] == ["kb-support"]


async def test_create_conflict(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/resources", json=llm_resource_body())).status_code == 201
    assert (await client.post("/api/v1/resources", json=llm_resource_body())).status_code == 409


async def test_update_and_delete(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    updated_body = {**llm_resource_body(), "default_model": "gpt-6"}
    updated = await client.put("/api/v1/resources/default-llm", json=updated_body)
    assert updated.status_code == 200
    assert updated.json()["default_model"] == "gpt-6"
    assert (await client.delete("/api/v1/resources/kb-support")).status_code == 204
    assert (await client.get("/api/v1/resources/kb-support")).status_code == 404


async def test_delete_refused_while_referenced(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    defn = load_example("echo-agent.yaml")
    created = await client.post("/api/v1/definitions", json=defn.canonical_dict())
    assert created.status_code == 201, created.text
    response = await client.delete("/api/v1/resources/default-llm")
    assert response.status_code == 409
    assert "echo-agent" in response.text


async def test_missing_resource_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/v1/resources/ghost")).status_code == 404
    assert (await client.delete("/api/v1/resources/ghost")).status_code == 404
    response = await client.put(
        "/api/v1/resources/ghost", json={**vector_db_body(), "name": "ghost"}
    )
    assert response.status_code == 404
