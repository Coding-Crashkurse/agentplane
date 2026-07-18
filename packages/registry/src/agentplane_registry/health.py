"""Background health job (SPEC §5.3).

Transitions ``starting -> healthy <-> unhealthy`` are made only here.
Agents: fetch the agent card THROUGH the gateway (the stored URL *is* the
gateway URL). MCP servers: ``tools/list`` via streamable HTTP when
``HEALTH_MCP=true``, else ``unknown``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import delete, select

from agentplane_core import agent_card_from_dict
from agentplane_registry.db import Database, EntryRow, EntryStatusEventRow, record_status_event
from agentplane_registry.settings import RegistrySettings

logger = logging.getLogger(__name__)

FAILURES_BEFORE_UNHEALTHY = 3
STARTING_RETRY_DELAY_S = 5.0
STARTING_RETRY_ATTEMPTS = 6


async def check_agent(url: str, *, timeout: float) -> bool:
    """2xx + parseable agent card -> healthy."""
    card_url = url.rstrip("/") + "/.well-known/agent-card.json"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(card_url)
        if not response.is_success:
            return False
        agent_card_from_dict(response.json())
    except Exception:
        return False
    return True


async def check_mcp_server(url: str, *, timeout: float) -> bool:
    """Initialize + tools/list via streamable HTTP (FastMCP client)."""
    from fastmcp import Client  # noqa: PLC0415 - deferred: heavy import

    try:
        async with asyncio.timeout(timeout):
            client = Client(url)
            async with client:
                await client.list_tools()
    except Exception:
        return False
    return True


class HealthJob:
    """Periodic checker with fast retry for fresh ``starting`` entries."""

    def __init__(self, db: Database, settings: RegistrySettings) -> None:
        self._db = db
        self._settings = settings
        self._failures: dict[str, int] = {}
        self._task: asyncio.Task[None] | None = None
        self._kicks: set[asyncio.Task[None]] = set()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="registry-health-job")

    def kick(self, entry_id: str) -> None:
        """Immediately fast-check one entry (e.g. just registered/re-enabled).

        Without this, a fresh "starting" entry sits until the next periodic
        pass picks it up (up to interval + jitter). No-op while the job is
        not running (health checking disabled, tests).
        """
        if self._task is None:
            return
        task = asyncio.create_task(self._check_starting(entry_id), name=f"health-kick-{entry_id}")
        self._kicks.add(task)
        task.add_done_callback(self._kicks.discard)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("health job pass failed")
            interval = self._settings.health_interval_s
            await asyncio.sleep(interval + random.uniform(0, interval * 0.1))

    async def _check(self, row: EntryRow) -> bool | None:
        """True/False = healthy/unhealthy signal; None = unknown (not checkable)."""
        if row.kind == "agent":
            return await check_agent(row.url, timeout=self._settings.health_timeout_s)
        if self._settings.health_mcp:
            return await check_mcp_server(row.url, timeout=self._settings.health_timeout_s)
        return None

    async def run_once(self) -> None:
        """One health pass over all enabled entries (checks run concurrently)."""
        await self._prune_history()
        async with self._db.session() as session:
            rows = (
                (await session.execute(select(EntryRow).where(EntryRow.enabled.is_(True))))
                .scalars()
                .all()
            )
        tasks = [
            self._check_starting(row.id)
            if row.status == "starting"
            else self._check_and_apply(row.id)
            for row in rows
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def _check_and_apply(self, entry_id: str) -> None:
        await self._apply(entry_id, await self._check_row(entry_id))

    async def _check_row(self, entry_id: str) -> bool | None:
        async with self._db.session() as session:
            row = await session.get(EntryRow, entry_id)
        if row is None:
            return None
        return await self._check(row)

    async def _check_starting(self, entry_id: str) -> None:
        """Fast retry (5s x6) so fresh deployments turn healthy quickly."""
        for attempt in range(STARTING_RETRY_ATTEMPTS):
            healthy = await self._check_row(entry_id)
            if healthy is None:
                await self._set_status(entry_id, "unknown")
                return
            if healthy:
                await self._set_status(entry_id, "healthy", touch_last_seen=True)
                return
            if attempt < STARTING_RETRY_ATTEMPTS - 1:
                await asyncio.sleep(STARTING_RETRY_DELAY_S)
        await self._set_status(entry_id, "unhealthy")

    async def _apply(self, entry_id: str, healthy: bool | None) -> None:
        if healthy is None:
            await self._set_status(entry_id, "unknown")
            self._failures.pop(entry_id, None)
            return
        if healthy:
            self._failures.pop(entry_id, None)
            await self._set_status(entry_id, "healthy", touch_last_seen=True)
            return
        failures = self._failures.get(entry_id, 0) + 1
        self._failures[entry_id] = failures
        if failures >= FAILURES_BEFORE_UNHEALTHY:
            await self._set_status(entry_id, "unhealthy")

    async def _set_status(
        self, entry_id: str, status: str, *, touch_last_seen: bool = False
    ) -> None:
        async with self._db.session() as session, session.begin():
            row = await session.get(EntryRow, entry_id)
            if row is None or not row.enabled:
                # Disabled mid-check (e.g. during the starting fast-retry
                # loop): keep the API-set "unknown", drop this result.
                return
            if row.status != status:
                record_status_event(session, entry_id, status)
            row.status = status
            if touch_last_seen:
                row.last_seen = datetime.now(UTC)

    async def _prune_history(self) -> None:
        """Drop status events past the retention window (SPEC §5.3)."""
        cutoff = datetime.now(UTC) - timedelta(hours=self._settings.history_retention_h)
        async with self._db.session() as session, session.begin():
            await session.execute(
                delete(EntryStatusEventRow).where(EntryStatusEventRow.at < cutoff)
            )


__all__ = ["HealthJob", "check_agent", "check_mcp_server"]
