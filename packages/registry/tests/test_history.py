"""Status history: transition recording, window query, retention pruning."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
from sqlalchemy import select

from agentplane_registry.db import Database, EntryRow, EntryStatusEventRow
from agentplane_registry.health import HealthJob

from .conftest import AGENT_CARD_JSON, agent_entry_body, make_settings


async def test_register_and_toggle_record_events(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/agents", json=agent_entry_body())
    entry_id = response.json()["id"]

    await client.put(f"/api/v1/agents/{entry_id}", json={"enabled": False})
    await client.put(f"/api/v1/agents/{entry_id}", json={"enabled": True})

    response = await client.get(f"/api/v1/agents/{entry_id}/history")
    assert response.status_code == 200
    body = response.json()
    assert [item["status"] for item in body["items"]] == ["starting", "unknown", "starting"]
    assert body["window_h"] == 24.0
    assert body["retention_h"] == 168.0


async def test_history_includes_state_at_window_start(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/agents", json=agent_entry_body())
    entry_id = response.json()["id"]

    # Age the registration event out of a tiny window: it must still be
    # returned as the state in effect when the window started.
    app_db: Database = client.app.state.registry.db  # type: ignore[attr-defined]
    async with app_db.session() as session, session.begin():
        rows = (
            (
                await session.execute(
                    select(EntryStatusEventRow).where(EntryStatusEventRow.entry_id == entry_id)
                )
            )
            .scalars()
            .all()
        )
        assert rows  # the registration event exists
        for event in rows:
            event.at = datetime.now(UTC) - timedelta(hours=2)

    response = await client.get(f"/api/v1/agents/{entry_id}/history", params={"hours": 1})
    body = response.json()
    assert [item["status"] for item in body["items"]] == ["starting"]


async def test_history_of_unknown_entry_is_404(client: httpx.AsyncClient) -> None:
    response = await client.get(f"/api/v1/agents/{uuid.uuid4()}/history")
    assert response.status_code == 404


async def test_delete_removes_events(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/agents", json=agent_entry_body())
    entry_id = response.json()["id"]
    await client.delete(f"/api/v1/agents/{entry_id}")
    response = await client.get(f"/api/v1/agents/{entry_id}/history")
    assert response.status_code == 404


@pytest.fixture
async def db() -> Database:
    database = Database("sqlite+aiosqlite://")
    await database.create_all()
    return database


async def _insert_agent(db: Database, *, status: str = "starting") -> str:
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
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    async with db.session() as session, session.begin():
        session.add(row)
    return row.id


async def _events(db: Database, entry_id: str) -> list[str]:
    async with db.session() as session:
        rows = (
            (
                await session.execute(
                    select(EntryStatusEventRow)
                    .where(EntryStatusEventRow.entry_id == entry_id)
                    .order_by(EntryStatusEventRow.at)
                )
            )
            .scalars()
            .all()
        )
    return [row.status for row in rows]


@respx.mock
async def test_health_job_records_transitions_not_samples(db: Database) -> None:
    respx.get("https://api.example/a2a/echo-agent/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json=AGENT_CARD_JSON)
    )
    entry_id = await _insert_agent(db, status="healthy")
    job = HealthJob(db, make_settings())
    await job.run_once()
    await job.run_once()
    # already healthy: repeated passes must not append "healthy" samples
    assert await _events(db, entry_id) == []


async def test_prune_drops_events_past_retention(db: Database) -> None:
    entry_id = await _insert_agent(db)
    old = datetime.now(UTC) - timedelta(hours=200)
    async with db.session() as session, session.begin():
        session.add(EntryStatusEventRow(entry_id=entry_id, status="healthy", at=old))
        session.add(EntryStatusEventRow(entry_id=entry_id, status="unhealthy"))
    job = HealthJob(db, make_settings(health_mcp=False))
    await job._prune_history()
    assert await _events(db, entry_id) == ["unhealthy"]
