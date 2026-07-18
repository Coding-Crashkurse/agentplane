"""Persistence layer (SPEC §5.2): SQLAlchemy async, SQLite by default."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, String, UniqueConstraint, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from agentplane_core import (
    EntryKind,
    HealthStatus,
    JsonObject,
    RegistryEntry,
    serialize_card,
)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EntryRow(Base):
    __tablename__ = "entries"
    __table_args__ = (UniqueConstraint("owner", "name", name="uq_entries_owner_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    owner: Mapped[str] = mapped_column(String(255), index=True)
    group: Mapped[str] = mapped_column(String(255), default="", index=True)
    url: Mapped[str] = mapped_column(String(2048))
    card_json: Mapped[str] = mapped_column(String)
    tags_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="starting", index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class EntryEmbeddingRow(Base):
    """Entry embedding vector.

    SPEC §5.2 requires this table only under [semantic]+Postgres; we persist
    JSON vectors on every backend so SQLite restarts do not re-embed, and use
    an in-process numpy brute-force search (fine <= a few thousand entries).
    """

    __tablename__ = "entry_embeddings"

    entry_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    vector: Mapped[list[float]] = mapped_column(JSON)
    norm: Mapped[float] = mapped_column(Float, default=1.0)


def row_to_entry(row: EntryRow) -> RegistryEntry:
    card: JsonObject = json.loads(row.card_json)
    return RegistryEntry.model_validate(
        {
            "id": uuid.UUID(row.id),
            "kind": row.kind,
            "card": card,
            "url": row.url,
            "tags": list(row.tags_json),
            "owner": row.owner,
            "group": row.group,
            "status": row.status,
            "enabled": row.enabled,
            "last_seen": row.last_seen,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
    )


def entry_kind(value: str) -> EntryKind:
    if value not in ("agent", "mcp_server"):
        raise ValueError(f"invalid entry kind {value!r}")
    return "agent" if value == "agent" else "mcp_server"


def health_status(value: str) -> HealthStatus:
    if value not in ("starting", "healthy", "unhealthy", "unknown"):
        raise ValueError(f"invalid health status {value!r}")
    result: HealthStatus = value  # type: ignore[assignment]  # narrowed by the check above
    return result


def card_to_json_str(entry: RegistryEntry) -> str:
    return json.dumps(serialize_card(entry.card), sort_keys=True)


class Database:
    """Engine + session factory wrapper."""

    def __init__(self, db_url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(db_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    def session(self) -> AsyncSession:
        return self.session_factory()


async def get_row(session: AsyncSession, entry_id: str) -> EntryRow | None:
    result = await session.execute(select(EntryRow).where(EntryRow.id == entry_id))
    return result.scalar_one_or_none()


__all__ = [
    "Base",
    "Database",
    "EntryEmbeddingRow",
    "EntryRow",
    "card_to_json_str",
    "entry_kind",
    "get_row",
    "health_status",
    "row_to_entry",
]
