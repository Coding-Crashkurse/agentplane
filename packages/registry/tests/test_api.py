"""Registry REST API behavior (SPEC §5.1)."""

from __future__ import annotations

import httpx

from .conftest import agent_entry_body, mcp_entry_body


async def test_register_returns_201_with_starting_status(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/agents", json=agent_entry_body())
    assert response.status_code == 201
    entry = response.json()
    assert entry["status"] == "starting"
    assert entry["owner"] == "anonymous"
    assert entry["card"]["name"] == "echo-agent"
    assert entry["url"] == "https://api.example/a2a/echo-agent"


async def test_register_conflict_on_same_owner_and_name(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/agents", json=agent_entry_body())).status_code == 201
    response = await client.post("/api/v1/agents", json=agent_entry_body())
    assert response.status_code == 409


async def test_register_rejects_private_url(client: httpx.AsyncClient) -> None:
    body = agent_entry_body()
    body["url"] = "http://runtime:8000/a2a/echo-agent"
    response = await client.post("/api/v1/agents", json=body)
    assert response.status_code == 422


async def test_get_and_delete_entry(client: httpx.AsyncClient) -> None:
    created = (await client.post("/api/v1/agents", json=agent_entry_body())).json()
    entry_id = created["id"]
    fetched = await client.get(f"/api/v1/agents/{entry_id}")
    assert fetched.status_code == 200
    assert (await client.delete(f"/api/v1/agents/{entry_id}")).status_code == 204
    assert (await client.get(f"/api/v1/agents/{entry_id}")).status_code == 404


async def test_update_entry_tags_and_url(client: httpx.AsyncClient) -> None:
    created = (await client.post("/api/v1/agents", json=agent_entry_body())).json()
    response = await client.put(
        f"/api/v1/agents/{created['id']}",
        json={"tags": ["updated"], "url": "https://api.example/a2a/renamed"},
    )
    assert response.status_code == 200
    updated = response.json()
    assert updated["tags"] == ["updated"]
    assert updated["url"].endswith("/renamed")


async def test_list_filters_kind_status_tags(client: httpx.AsyncClient) -> None:
    await client.post("/api/v1/agents", json=agent_entry_body("agent-a"))
    await client.post("/api/v1/agents", json=mcp_entry_body("rag-b"))
    everything = (await client.get("/api/v1/agents")).json()
    assert everything["total"] == 2
    agents_only = (await client.get("/api/v1/agents", params={"kind": "agent"})).json()
    assert agents_only["total"] == 1
    assert agents_only["items"][0]["kind"] == "agent"
    tagged = (await client.get("/api/v1/agents", params={"tags": "rag"})).json()
    assert tagged["total"] == 1
    starting = (await client.get("/api/v1/agents", params={"status": "starting"})).json()
    assert starting["total"] == 2


async def test_list_pagination(client: httpx.AsyncClient) -> None:
    for i in range(3):
        await client.post("/api/v1/agents", json=agent_entry_body(f"agent-{i}"))
    page = (await client.get("/api/v1/agents", params={"limit": 2, "offset": 2})).json()
    assert page["total"] == 3
    assert len(page["items"]) == 1
    assert page["limit"] == 2 and page["offset"] == 2


async def test_search_text_match(client: httpx.AsyncClient) -> None:
    await client.post("/api/v1/agents", json=agent_entry_body("echo-agent"))
    await client.post("/api/v1/agents", json=mcp_entry_body("support-rag"))
    hits = (await client.get("/api/v1/agents/search", params={"q": "knowledge base"})).json()
    assert hits["total"] == 1
    assert hits["items"][0]["card"]["name"] == "support-rag"
    skill_hits = (await client.get("/api/v1/agents/search", params={"q": "answer"})).json()
    assert skill_hits["total"] == 1
    assert skill_hits["items"][0]["card"]["name"] == "echo-agent"


async def test_search_semantic_degrades_without_extra(client: httpx.AsyncClient) -> None:
    await client.post("/api/v1/agents", json=agent_entry_body())
    response = await client.get("/api/v1/agents/search", params={"q": "echo", "semantic": "true"})
    assert response.status_code == 200
    # no embeddings configured -> degraded to text search, announced via header
    assert response.headers.get("X-Degraded") == "semantic"
    assert response.json()["total"] == 1


async def test_capabilities(client: httpx.AsyncClient) -> None:
    caps = (await client.get("/api/v1/capabilities")).json()
    assert caps == {"semantic_search": False, "auth": "none", "version": caps["version"]}


async def test_healthz_and_readyz(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/readyz")).status_code == 200
