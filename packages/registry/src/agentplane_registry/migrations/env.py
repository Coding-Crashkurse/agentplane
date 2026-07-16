"""Alembic environment: async-capable, no ``alembic.ini`` (SPEC §5.2).

Normally driven programmatically by ``agentplane_registry.migrate`` with a live
connection in ``config.attributes["connection"]`` (the official
connection-sharing recipe). Without one — e.g. ``alembic revision`` from a
checkout — it falls back to an async engine built from ``sqlalchemy.url``, so
both ``sqlite+aiosqlite`` and ``postgresql+asyncpg`` URLs work.
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from agentplane_registry.db import Base
from agentplane_registry.migrate import VERSION_TABLE

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running against a database."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, version_table=VERSION_TABLE
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("no connection provided and no sqlalchemy.url configured")
    connectable = create_async_engine(url, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    connection: Connection | None = config.attributes.get("connection")
    if connection is None:
        asyncio.run(run_async_migrations())
    else:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
