"""Startup migrations (SPEC §6): fresh DB, legacy pre-Alembic DB, idempotency."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import func, inspect, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from agentplane_runtime.db import Base, DefinitionRow
from agentplane_runtime.migrate import (
    BASELINE_REVISION,
    VERSION_TABLE,
    alembic_config,
    run_migrations,
)


def _db_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{(tmp_path / 'runtime.db').as_posix()}"


def _head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    assert head is not None
    return head


def _current_revision(conn: Connection) -> str | None:
    ctx = MigrationContext.configure(conn, opts={"version_table": VERSION_TABLE})
    return ctx.get_current_revision()


def _column_names(conn: Connection, table: str) -> set[str]:
    return {col["name"] for col in inspect(conn).get_columns(table)}


def _schema_snapshot(conn: Connection) -> dict[str, tuple[dict[str, bool], set[str | None]]]:
    """Tables -> (column nullability, index names); the version table excluded."""
    inspector = inspect(conn)
    return {
        table: (
            {col["name"]: col["nullable"] for col in inspector.get_columns(table)},
            {idx["name"] for idx in inspector.get_indexes(table)},
        )
        for table in inspector.get_table_names()
        if table != VERSION_TABLE
    }


def _create_legacy_schema(conn: Connection) -> None:
    """The true 0.0.2 database: baseline DDL (no ``group``), no Alembic version table."""
    command.upgrade(alembic_config(conn), BASELINE_REVISION)
    conn.execute(text(f"DROP TABLE {VERSION_TABLE}"))
    now = datetime.now(UTC).isoformat()
    conn.execute(
        text(
            "INSERT INTO definitions (name, owner, status, draft_json, created_at, updated_at)"
            " VALUES (:name, :owner, :status, :draft_json, :created_at, :updated_at)"
        ),
        {
            "name": "echo-agent",
            "owner": "anonymous",
            "status": "draft",
            "draft_json": "{}",
            "created_at": now,
            "updated_at": now,
        },
    )


async def test_fresh_database_migrates_to_head_and_matches_create_all(tmp_path: Path) -> None:
    engine = create_async_engine(_db_url(tmp_path))
    reference = create_async_engine("sqlite+aiosqlite://")
    try:
        await run_migrations(engine)
        async with engine.connect() as conn:
            revision = await conn.run_sync(_current_revision)
            migrated = await conn.run_sync(_schema_snapshot)
        assert revision == _head_revision()
        assert {"definitions", "definition_versions", "resources", "secrets"} <= set(migrated)

        # the baseline must create exactly what the models' create_all creates
        async with reference.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with reference.connect() as conn:
            expected = await conn.run_sync(_schema_snapshot)
        assert migrated == expected
    finally:
        await engine.dispose()
        await reference.dispose()


async def test_legacy_database_is_stamped_then_upgraded_and_data_survives(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(_db_url(tmp_path))
    try:
        async with engine.begin() as conn:
            await conn.run_sync(_create_legacy_schema)
        async with engine.connect() as conn:
            assert await conn.run_sync(_current_revision) is None
            legacy_columns = await conn.run_sync(_column_names, "definitions")
        assert "group" not in legacy_columns  # 0.0.2 predates team sharing

        await run_migrations(engine)

        async with engine.connect() as conn:
            revision = await conn.run_sync(_current_revision)
            count = (
                await conn.execute(select(func.count()).select_from(DefinitionRow))
            ).scalar_one()
            group = (await conn.execute(select(DefinitionRow.group))).scalar_one()
            resource_columns = await conn.run_sync(_column_names, "resources")
        assert revision == _head_revision()
        assert count == 1
        assert group == ""  # backfilled by the group migration
        assert "group" in resource_columns
    finally:
        await engine.dispose()


async def test_second_run_is_idempotent(tmp_path: Path) -> None:
    engine = create_async_engine(_db_url(tmp_path))
    try:
        await run_migrations(engine)
        await run_migrations(engine)
        async with engine.connect() as conn:
            assert await conn.run_sync(_current_revision) == _head_revision()
    finally:
        await engine.dispose()
