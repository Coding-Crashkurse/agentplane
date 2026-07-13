"""Definition lifecycle service (SPEC §6.1/§6.2): draft -> versions -> endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from agentplane_core import (
    DefinitionInfo,
    DeploymentInfo,
    FlowDefinition,
    ValidationResult,
)
from agentplane_runtime.db import (
    Database,
    DefinitionRow,
    VersionRow,
    dump_definition,
    latest_version,
    load_definition,
)
from agentplane_runtime.registration import RegistryRegistrar
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.serving import EndpointManager
from agentplane_runtime.validation import validate_full


class DefinitionNotFoundError(Exception):
    pass


class DefinitionConflictError(Exception):
    pass


class DefinitionStateError(Exception):
    """Operation not allowed in the current lifecycle state."""


class DefinitionInvalidError(Exception):
    def __init__(self, result: ValidationResult) -> None:
        super().__init__("definition validation failed")
        self.result = result


class DefinitionService:
    """Owns definitions and their lifecycle: draft, deploy, undeploy, delete."""

    def __init__(
        self,
        db: Database,
        resources: ResourceService,
        endpoints: EndpointManager,
        registrar: RegistryRegistrar,
    ) -> None:
        self._db = db
        self._resources = resources
        self._endpoints = endpoints
        self._registrar = registrar

    async def validate(self, defn: FlowDefinition | dict[str, object]) -> ValidationResult:
        return await validate_full(defn, self._resources)

    async def _row(self, name: str) -> DefinitionRow:
        async with self._db.session() as session:
            row = await session.get(DefinitionRow, name)
        if row is None:
            raise DefinitionNotFoundError(name)
        return row

    async def create_draft(self, defn: FlowDefinition, owner: str) -> DefinitionInfo:
        result = await self.validate(defn)
        if not result.valid:
            raise DefinitionInvalidError(result)
        async with self._db.session() as session:
            existing = await session.get(DefinitionRow, defn.name)
        if existing is not None:
            raise DefinitionConflictError(defn.name)
        now = datetime.now(UTC)
        row = DefinitionRow(
            name=defn.name,
            owner=owner,
            status="draft",
            draft_json=dump_definition(defn),
            created_at=now,
            updated_at=now,
        )
        async with self._db.session() as session, session.begin():
            session.add(row)
        return await self.info(defn.name)

    async def update_draft(self, name: str, defn: FlowDefinition) -> DefinitionInfo:
        if defn.name != name:
            raise DefinitionConflictError(
                f"definition name {defn.name!r} does not match path {name!r}"
            )
        result = await self.validate(defn)
        if not result.valid:
            raise DefinitionInvalidError(result)
        async with self._db.session() as session, session.begin():
            row = await session.get(DefinitionRow, name)
            if row is None:
                raise DefinitionNotFoundError(name)
            row.draft_json = dump_definition(defn)
            row.updated_at = datetime.now(UTC)
        return await self.info(name)

    async def deploy(
        self, name: str, *, version: int | None = None, ephemeral: bool = False
    ) -> DeploymentInfo:
        row = await self._row(name)
        if ephemeral:
            return await self._deploy_ephemeral(row)

        if version is None:
            draft = load_definition(row.draft_json)
            result = await self.validate(draft)
            if not result.valid:
                raise DefinitionInvalidError(result)
            async with self._db.session() as session, session.begin():
                new_version = await latest_version(session, name) + 1
                session.add(
                    VersionRow(
                        name=name,
                        version=new_version,
                        definition_json=dump_definition(draft),
                    )
                )
            defn, active_version = draft, new_version
        else:
            defn, active_version = await self._load_version(name, version), version

        endpoint = await self._endpoints.start(defn, active_version)
        registry_id = await self._registrar.register(
            defn, endpoint.public_url, uuid.UUID(row.registry_id) if row.registry_id else None
        )
        async with self._db.session() as session, session.begin():
            fresh = await session.get(DefinitionRow, name)
            if fresh is not None:
                fresh.status = "deployed"
                fresh.deployed_version = active_version
                fresh.registry_id = str(registry_id) if registry_id else fresh.registry_id
                fresh.updated_at = datetime.now(UTC)
        return DeploymentInfo(
            name=name,
            version=active_version,
            endpoint_url=endpoint.public_url,
            registry_id=registry_id,
        )

    async def _deploy_ephemeral(self, row: DefinitionRow) -> DeploymentInfo:
        """Playground endpoint for the draft: no registry entry, TTL-bound (§6.2)."""
        draft = load_definition(row.draft_json)
        result = await self.validate(draft)
        if not result.valid:
            raise DefinitionInvalidError(result)
        endpoint = await self._endpoints.start(draft, 0, ephemeral=True)
        return DeploymentInfo(
            name=row.name, version=0, endpoint_url=endpoint.public_url, registry_id=None
        )

    async def _load_version(self, name: str, version: int) -> FlowDefinition:
        async with self._db.session() as session:
            result = await session.execute(
                select(VersionRow).where(VersionRow.name == name, VersionRow.version == version)
            )
            row = result.scalar_one_or_none()
        if row is None:
            raise DefinitionNotFoundError(f"{name} version {version}")
        return load_definition(row.definition_json)

    async def undeploy(self, name: str) -> None:
        row = await self._row(name)
        if row.status != "deployed":
            raise DefinitionStateError(f"{name} is not deployed")
        await self._endpoints.stop(name)
        await self._registrar.deregister(uuid.UUID(row.registry_id) if row.registry_id else None)
        async with self._db.session() as session, session.begin():
            fresh = await session.get(DefinitionRow, name)
            if fresh is not None:
                fresh.status = "undeployed"
                fresh.deployed_version = None
                fresh.registry_id = None
                fresh.updated_at = datetime.now(UTC)

    async def delete(self, name: str) -> None:
        row = await self._row(name)
        if row.status == "deployed":
            raise DefinitionStateError(f"{name} is deployed; undeploy first")
        async with self._db.session() as session, session.begin():
            fresh = await session.get(DefinitionRow, name)
            if fresh is not None:
                await session.delete(fresh)
            versions = (
                (await session.execute(select(VersionRow).where(VersionRow.name == name)))
                .scalars()
                .all()
            )
            for version_row in versions:
                await session.delete(version_row)

    async def info(self, name: str, *, include_definition: bool = False) -> DefinitionInfo:
        row = await self._row(name)
        return await self._to_info(row, include_definition=include_definition)

    async def _to_info(
        self, row: DefinitionRow, *, include_definition: bool = False
    ) -> DefinitionInfo:
        draft = load_definition(row.draft_json)
        async with self._db.session() as session:
            latest = await latest_version(session, row.name)
        endpoint = self._endpoints.endpoint_for(row.name)
        status = row.status
        if status not in ("draft", "deployed", "undeployed"):  # pragma: no cover
            status = "draft"
        return DefinitionInfo.model_validate(
            {
                "name": row.name,
                "display_name": draft.display_name,
                "description": draft.description,
                "tags": draft.tags,
                "expose_kind": draft.expose.kind,
                "status": status,
                "latest_version": latest or None,
                "deployed_version": row.deployed_version,
                "endpoint_url": endpoint.public_url if endpoint is not None else None,
                "owner": row.owner,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "definition": draft if include_definition else None,
            }
        )

    async def list(self, status: str | None = None) -> list[DefinitionInfo]:
        stmt = select(DefinitionRow).order_by(DefinitionRow.name)
        if status is not None:
            stmt = stmt.where(DefinitionRow.status == status)
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [await self._to_info(row) for row in rows]

    async def export(self, name: str, version: int | None = None) -> FlowDefinition:
        if version is not None:
            return await self._load_version(name, version)
        row = await self._row(name)
        return load_definition(row.draft_json)

    async def restore_deployed_endpoints(self) -> None:
        """On startup: restart endpoints for deployed definitions and re-register."""
        async with self._db.session() as session:
            rows = (
                (
                    await session.execute(
                        select(DefinitionRow).where(DefinitionRow.status == "deployed")
                    )
                )
                .scalars()
                .all()
            )
        for row in rows:
            if row.deployed_version is None:
                continue
            defn = await self._load_version(row.name, row.deployed_version)
            endpoint = await self._endpoints.start(defn, row.deployed_version)
            await self._registrar.register(
                defn,
                endpoint.public_url,
                uuid.UUID(row.registry_id) if row.registry_id else None,
            )


__all__ = [
    "DefinitionConflictError",
    "DefinitionInvalidError",
    "DefinitionNotFoundError",
    "DefinitionService",
    "DefinitionStateError",
]
