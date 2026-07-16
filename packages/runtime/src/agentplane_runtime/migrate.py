"""Programmatic Alembic migrations (SPEC §6): the schema is migration-managed.

No ``alembic.ini`` — the Config is built from the ``migrations/`` directory
bundled inside the package. Startup calls :func:`run_migrations` instead of
``create_all``:

- empty database → upgrade to head (the migration chain creates the full schema)
- pre-Alembic database (tables exist, but no version table) → stamp the
  baseline revision, then upgrade to head
- migrated database → upgrade to head (no-op when already current)

The version table carries a service suffix so registry and runtime can share
one database without clashing. Deliberately a sibling implementation of the
registry's ``migrate.py`` — services never import each other (invariant 3).
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

BASELINE_REVISION = "0001"
VERSION_TABLE = "alembic_version_runtime"

_MARKER_TABLE = "definitions"  # present in every pre-Alembic runtime database


def alembic_config(connection: Connection | None = None) -> Config:
    """Alembic Config pointing at the bundled migrations; no ini file involved."""
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def _migrate(connection: Connection) -> None:
    cfg = alembic_config(connection)
    tables = set(inspect(connection).get_table_names())
    if _MARKER_TABLE in tables and VERSION_TABLE not in tables:
        command.stamp(cfg, BASELINE_REVISION)
    command.upgrade(cfg, "head")


async def run_migrations(engine: AsyncEngine) -> None:
    """Bring the database schema to the latest revision (see module docstring)."""
    async with engine.begin() as conn:
        await conn.run_sync(_migrate)


__all__ = ["BASELINE_REVISION", "VERSION_TABLE", "alembic_config", "run_migrations"]
