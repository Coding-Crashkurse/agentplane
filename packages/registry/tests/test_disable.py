"""Soft-disable of registry entries: API toggle, discovery and health behavior."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import httpx
import pytest
import respx

from agentplane_registry.db import Database, EntryRow
from agentplane_registry.health import HealthJob

from .conftest import AGENT_CARD_JSON, agent_entry_body, make_settings


async def _register(client: httpx.AsyncClient, name: str = "echo-agent") -> str:
    response = await client.post("/api/v1/agents", json=agent_entry_body(name))
    assert response.status_code == 201
    entry_id: str = response.json()["id"]
    return entry_id


async def test_disable_and_reenable_entry(client: httpx.AsyncClient) -> None:
    entry_id = await _register(client)

    response = await client.put(f"/api/v1/agents/{entry_id}", json={"enabled": False})
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["status"] == "unknown"

    response = await client.put(f"/api/v1/agents/{entry_id}", json={"enabled": True})
    body = response.json()
    assert body["enabled"] is True
    assert body["status"] == "starting"


async def test_noop_patch_keeps_status(client: httpx.AsyncClient) -> None:
    entry_id = await _register(client)
    response = await client.put(f"/api/v1/agents/{entry_id}", json={"enabled": True})
    assert response.json()["status"] == "starting"
    response = await client.put(f"/api/v1/agents/{entry_id}", json={"tags": ["new"]})
    assert response.json()["enabled"] is True


async def test_list_shows_disabled_and_filters(client: httpx.AsyncClient) -> None:
    kept = await _register(client, "kept-agent")
    disabled = await _register(client, "disabled-agent")
    await client.put(f"/api/v1/agents/{disabled}", json={"enabled": False})

    response = await client.get("/api/v1/agents")
    ids = {item["id"] for item in response.json()["items"]}
    assert ids == {kept, disabled}

    response = await client.get("/api/v1/agents", params={"enabled": False})
    assert [item["id"] for item in response.json()["items"]] == [disabled]

    response = await client.get("/api/v1/agents", params={"enabled": True})
    assert [item["id"] for item in response.json()["items"]] == [kept]


async def test_search_excludes_disabled_by_default(client: httpx.AsyncClient) -> None:
    await _register(client, "kept-agent")
    disabled = await _register(client, "disabled-agent")
    await client.put(f"/api/v1/agents/{disabled}", json={"enabled": False})

    response = await client.get("/api/v1/agents/search", params={"q": "agent"})
    names = {item["card"]["name"] for item in response.json()["items"]}
    assert names == {"kept-agent"}

    response = await client.get(
        "/api/v1/agents/search", params={"q": "agent", "include_disabled": True}
    )
    names = {item["card"]["name"] for item in response.json()["items"]}
    assert names == {"kept-agent", "disabled-agent"}


@pytest.fixture
async def db() -> Database:
    database = Database("sqlite+aiosqlite://")
    await database.create_all()
    return database


async def _insert_agent(db: Database, *, status: str, enabled: bool) -> str:
    body = agent_entry_body()
    row = EntryRow(
        id=str(uuid.uuid4()),
        kind="agent",
        name="echo-agent",
        owner="anonymous",
        url=body["url"],
        card_json=json.dumps(body["card"]),
        tags_json=["demo"],
        status=status,
        enabled=enabled,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    async with db.session() as session, session.begin():
        session.add(row)
    return row.id


async def _row_status(db: Database, entry_id: str) -> str:
    async with db.session() as session:
        row = await session.get(EntryRow, entry_id)
    assert row is not None
    return row.status


@respx.mock
async def test_health_job_skips_disabled_entries(db: Database) -> None:
    respx.get("https://api.example/a2a/echo-agent/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json=AGENT_CARD_JSON)
    )
    entry_id = await _insert_agent(db, status="unknown", enabled=False)
    job = HealthJob(db, make_settings())
    await job.run_once()
    assert await _row_status(db, entry_id) == "unknown"


async def test_set_status_ignores_disabled_rows(db: Database) -> None:
    """A check finishing after a concurrent disable must not overwrite it."""
    entry_id = await _insert_agent(db, status="unknown", enabled=False)
    job = HealthJob(db, make_settings())
    await job._set_status(entry_id, "healthy", touch_last_seen=True)
    assert await _row_status(db, entry_id) == "unknown"
