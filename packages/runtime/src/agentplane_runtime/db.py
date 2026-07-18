"""Runtime persistence: definitions (draft + immutable versions), resources, secrets."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import TypeAdapter
from sqlalchemy import DateTime, Integer, String, UniqueConstraint, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from agentplane_core import FlowDefinition, Resource

RESOURCE_ADAPTER: TypeAdapter[Resource] = TypeAdapter(Resource)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DefinitionRow(Base):
    __tablename__ = "definitions"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(255), index=True)
    owner_name: Mapped[str] = mapped_column(String(255), default="")
    group: Mapped[str] = mapped_column(String(255), default="", index=True)
    status: Mapped[str] = mapped_column(String(16), default="draft", index=True)
    draft_json: Mapped[str] = mapped_column(String)
    deployed_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    registry_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class VersionRow(Base):
    __tablename__ = "definition_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_versions_name_version"),
        UniqueConstraint("name", "version_label", name="uq_versions_name_label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    # Publisher-chosen semantic version for this snapshot; unique per definition.
    # NULL for versions deployed without a label (SQL treats NULLs as distinct).
    version_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    definition_json: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ResourceRow(Base):
    __tablename__ = "resources"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    owner: Mapped[str] = mapped_column(String(255), index=True)
    group: Mapped[str] = mapped_column(String(255), default="", index=True)
    config_json: Mapped[str] = mapped_column(String)  # secrets replaced by refs
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class SecretRow(Base):
    __tablename__ = "secrets"

    ref: Mapped[str] = mapped_column(String(255), primary_key=True)
    ciphertext: Mapped[str] = mapped_column(String)


def dump_definition(defn: FlowDefinition) -> str:
    return json.dumps(defn.canonical_dict(), sort_keys=False)


def load_definition(raw: str) -> FlowDefinition:
    return FlowDefinition.model_validate(json.loads(raw))


def load_resource(row: ResourceRow) -> Resource:
    return RESOURCE_ADAPTER.validate_python(json.loads(row.config_json))


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


async def latest_version(session: AsyncSession, name: str) -> int:
    result = await session.execute(
        select(VersionRow.version)
        .where(VersionRow.name == name)
        .order_by(VersionRow.version.desc())
    )
    first = result.scalars().first()
    return first or 0


async def deployed_count(session: AsyncSession, owner: str) -> int:
    """How many non-ephemeral definitions this owner currently keeps deployed."""
    result = await session.execute(
        select(func.count())
        .select_from(DefinitionRow)
        .where(DefinitionRow.owner == owner, DefinitionRow.status == "deployed")
    )
    return result.scalar_one()


async def version_row(session: AsyncSession, name: str, version: int) -> VersionRow | None:
    result = await session.execute(
        select(VersionRow).where(VersionRow.name == name, VersionRow.version == version)
    )
    return result.scalar_one_or_none()


async def version_row_by_label(session: AsyncSession, name: str, label: str) -> VersionRow | None:
    result = await session.execute(
        select(VersionRow).where(VersionRow.name == name, VersionRow.version_label == label)
    )
    return result.scalar_one_or_none()


__all__ = [
    "RESOURCE_ADAPTER",
    "Base",
    "Database",
    "DefinitionRow",
    "ResourceRow",
    "SecretRow",
    "VersionRow",
    "deployed_count",
    "dump_definition",
    "latest_version",
    "load_definition",
    "load_resource",
    "version_row",
    "version_row_by_label",
]
