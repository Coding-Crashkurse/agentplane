"""Health job transitions (SPEC §5.3)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import httpx
import pytest
import respx

from agentplane_registry.db import Database, EntryRow
from agentplane_registry.health import HealthJob, check_agent

from .conftest import AGENT_CARD_JSON, agent_entry_body, make_settings


@pytest.fixture
async def db() -> Database:
    database = Database("sqlite+aiosqlite://")
    await database.create_all()
    return database


async def _insert_agent(db: Database, *, status: str = "healthy") -> str:
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


async def _status(db: Database, entry_id: str) -> str:
    async with db.session() as session:
        row = await session.get(EntryRow, entry_id)
    assert row is not None
    return row.status


@respx.mock
async def test_check_agent_healthy() -> None:
    respx.get("https://api.example/a2a/echo-agent/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json=AGENT_CARD_JSON)
    )
    assert await check_agent("https://api.example/a2a/echo-agent", timeout=5.0)


@respx.mock
async def test_check_agent_unparseable_card_is_unhealthy() -> None:
    respx.get("https://api.example/a2a/echo-agent/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json={"skills": "not-a-list"})
    )
    assert not await check_agent("https://api.example/a2a/echo-agent", timeout=5.0)


@respx.mock
async def test_healthy_entry_needs_three_failures_to_flip(db: Database) -> None:
    entry_id = await _insert_agent(db, status="healthy")
    respx.get("https://api.example/a2a/echo-agent/.well-known/agent-card.json").mock(
        return_value=httpx.Response(503)
    )
    job = HealthJob(db, make_settings(health_interval_s=0.01))
    await job.run_once()
    assert await _status(db, entry_id) == "healthy"  # 1st failure
    await job.run_once()
    assert await _status(db, entry_id) == "healthy"  # 2nd failure
    await job.run_once()
    assert await _status(db, entry_id) == "unhealthy"  # 3rd consecutive failure


@respx.mock
async def test_unhealthy_recovers_immediately(db: Database) -> None:
    entry_id = await _insert_agent(db, status="unhealthy")
    respx.get("https://api.example/a2a/echo-agent/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json=AGENT_CARD_JSON)
    )
    job = HealthJob(db, make_settings())
    await job.run_once()
    assert await _status(db, entry_id) == "healthy"
    async with db.session() as session:
        row = await session.get(EntryRow, entry_id)
    assert row is not None and row.last_seen is not None


@respx.mock
async def test_starting_entry_turns_healthy_fast(db: Database) -> None:
    entry_id = await _insert_agent(db, status="starting")
    respx.get("https://api.example/a2a/echo-agent/.well-known/agent-card.json").mock(
        return_value=httpx.Response(200, json=AGENT_CARD_JSON)
    )
    job = HealthJob(db, make_settings())
    await job.run_once()
    assert await _status(db, entry_id) == "healthy"


async def test_mcp_entry_without_health_mcp_is_unknown(db: Database) -> None:
    row = EntryRow(
        id=str(uuid.uuid4()),
        kind="mcp_server",
        name="support-rag",
        owner="anonymous",
        url="https://api.example/mcp/support-rag",
        card_json=json.dumps({"name": "support-rag", "url": "x", "tools": []}),
        tags_json=[],
        status="healthy",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    async with db.session() as session, session.begin():
        session.add(row)
    job = HealthJob(db, make_settings(health_mcp=False))
    await job.run_once()
    assert await _status(db, row.id) == "unknown"
