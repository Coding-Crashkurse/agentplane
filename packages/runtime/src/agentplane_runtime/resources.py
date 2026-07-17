"""Resource service (SPEC §6.3): CRUD, write-only secrets, E022 dimension check.

Create/update stay offline: a resource carries no collection, so there is
nothing to compare a dimension against until a retrieval node names one. The
E022 check therefore runs in definition validation (see ``validation.py``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import ValidationError
from sqlalchemy import or_, select

from agentplane_core import (
    FlowDefinition,
    LlmCallNode,
    McpToolNode,
    Resource,
    RetrievalNode,
    SecretsProvider,
    ValidationIssue,
    ValidationResult,
    VectorDBResource,
)
from agentplane_runtime.auth import AccessScope
from agentplane_runtime.db import (
    RESOURCE_ADAPTER,
    Database,
    DefinitionRow,
    ResourceRow,
    VersionRow,
    load_definition,
    load_resource,
)
from agentplane_runtime.vector import VectorDBError, reader_for

_SECRET_FIELDS = ("api_key_secret", "dsn_secret", "auth_secret")

# Definition validation reads the collection's vector size over the network
# (the established design — the dimension lives in the DB, not the definition),
# but a Validate call must never hang on an unreachable DB. This bounds that
# single probe; a timeout is treated exactly like "unreachable" -> no E022.
_VALIDATE_DIMENSION_TIMEOUT_S = 3.0


class ResourceConflictError(Exception):
    """Resource exists (create) or is still referenced (delete)."""


class ResourceNotFoundError(Exception):
    """No resource with that name."""


class ResourceValidationError(Exception):
    """Resource failed a stateful check (e.g. E022)."""

    def __init__(self, result: ValidationResult) -> None:
        super().__init__("resource validation failed")
        self.result = result


def secret_ref(resource_name: str, field: str) -> str:
    return f"resource/{resource_name}/{field}"


class ResourceService:
    """Owns resource rows; secret values live only in the SecretsProvider."""

    def __init__(self, db: Database, secrets: SecretsProvider) -> None:
        self._db = db
        self._secrets = secrets

    async def _store_secrets(self, resource: Resource) -> Resource:
        """Move secret values into the provider; keep placeholder refs in the row."""
        data = RESOURCE_ADAPTER.dump_python(resource, mode="json", context={"reveal_secrets": True})
        for field in _SECRET_FIELDS:
            value = data.get(field)
            if isinstance(value, str) and value:
                await self._secrets.put(secret_ref(resource.name, field), value)
                data[field] = secret_ref(resource.name, field)
        return RESOURCE_ADAPTER.validate_python(data)

    async def secret_value(self, resource_name: str, field: str) -> str:
        return await self._secrets.get(secret_ref(resource_name, field))

    async def check_collection_dimension(
        self, resource: VectorDBResource, collection: str, path: str
    ) -> ValidationIssue | None:
        """E022 for a concrete collection (SPEC §3.7/§6.3).

        The collection lives on the retrieval *node*, not on the resource, so
        this check only has something to compare during definition validation —
        which is why ``path`` points at the offending node. The resource's
        declared ``embedding.dimension`` is compared against the collection's
        actual vector size, read from the DB with a short bounded timeout. An
        unreachable vector DB (or a timeout) is not an E022: it surfaces at
        execution time, not as a schema error.
        """
        api_key, dsn = "", ""
        try:
            if resource.api_key_secret:
                api_key = await self.secret_value(resource.name, "api_key_secret")
            if resource.dsn_secret:
                dsn = await self.secret_value(resource.name, "dsn_secret")
        except KeyError:
            pass
        try:
            reader = await reader_for(
                resource, api_key=api_key, dsn=dsn, timeout=_VALIDATE_DIMENSION_TIMEOUT_S
            )
            actual = await reader.collection_dimension(collection)
        except VectorDBError:
            return None  # unreachable — runtime errors surface at execution time
        if actual != resource.embedding.dimension:
            return ValidationIssue(
                code="E022",
                severity="error",
                path=path,
                message=(
                    f"resource {resource.name!r} embeds with dimension "
                    f"{resource.embedding.dimension}, but collection {collection!r} has "
                    f"dimension {actual}"
                ),
            )
        return None

    async def create(self, resource: Resource, owner: str, group: str = "") -> Resource:
        async with self._db.session() as session:
            existing = await session.get(ResourceRow, resource.name)
        if existing is not None:
            raise ResourceConflictError(f"resource {resource.name!r} already exists")
        stored = await self._store_secrets(resource)
        now = datetime.now(UTC)
        row = ResourceRow(
            name=resource.name,
            kind=resource.kind,
            owner=owner,
            group=group,
            config_json=json.dumps(
                RESOURCE_ADAPTER.dump_python(stored, mode="json", context={"reveal_secrets": True})
            ),
            created_at=now,
            updated_at=now,
        )
        async with self._db.session() as session, session.begin():
            session.add(row)
        return self._redacted(stored)

    async def update(
        self, name: str, resource: Resource, scope: AccessScope | None = None
    ) -> Resource:
        """Update a resource; ``scope`` (when set) restricts to the caller's own/team."""
        if resource.name != name:
            raise ResourceValidationError(
                ValidationResult(
                    valid=False,
                    issues=[
                        ValidationIssue(
                            code="E011",
                            severity="error",
                            path="name",
                            message="resource name cannot be changed",
                        )
                    ],
                )
            )
        async with self._db.session() as session:
            row = await session.get(ResourceRow, name)
        if row is None or (scope is not None and not scope.allows(row.owner, row.group)):
            raise ResourceNotFoundError(name)
        stored = await self._store_secrets(resource)
        async with self._db.session() as session, session.begin():
            fresh = await session.get(ResourceRow, name)
            if fresh is None:
                raise ResourceNotFoundError(name)
            fresh.config_json = json.dumps(
                RESOURCE_ADAPTER.dump_python(stored, mode="json", context={"reveal_secrets": True})
            )
            fresh.updated_at = datetime.now(UTC)
        return self._redacted(stored)

    async def get(self, name: str, scope: AccessScope | None = None) -> Resource:
        async with self._db.session() as session:
            row = await session.get(ResourceRow, name)
        if row is None or (scope is not None and not scope.allows(row.owner, row.group)):
            raise ResourceNotFoundError(name)
        return self._redacted(load_resource(row))

    async def get_raw(self, name: str) -> Resource:
        """Resource with secret *refs* intact — engine-internal use only."""
        async with self._db.session() as session:
            row = await session.get(ResourceRow, name)
        if row is None:
            raise ResourceNotFoundError(name)
        return load_resource(row)

    async def list(
        self, kind: str | None = None, scope: AccessScope | None = None
    ) -> list[Resource]:
        stmt = select(ResourceRow).order_by(ResourceRow.name)
        if kind is not None:
            stmt = stmt.where(ResourceRow.kind == kind)
        if scope is not None and not scope.unrestricted:
            conditions = [ResourceRow.owner == scope.sub]
            if scope.groups:
                conditions.append(ResourceRow.group.in_(scope.groups))
            stmt = stmt.where(or_(*conditions))
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [self._redacted(load_resource(row)) for row in rows]

    async def delete(self, name: str, scope: AccessScope | None = None) -> None:
        async with self._db.session() as session:
            row = await session.get(ResourceRow, name)
        if row is None or (scope is not None and not scope.allows(row.owner, row.group)):
            raise ResourceNotFoundError(name)
        referencing = await self._referencing_definitions(name)
        if referencing:
            raise ResourceConflictError(
                f"resource {name!r} is referenced by: {', '.join(sorted(referencing))}"
            )
        async with self._db.session() as session, session.begin():
            row = await session.get(ResourceRow, name)
            if row is None:
                raise ResourceNotFoundError(name)
            await session.delete(row)
        for field in _SECRET_FIELDS:
            await self._secrets.delete(secret_ref(name, field))

    async def _referencing_definitions(self, resource_name: str) -> set[str]:
        names: set[str] = set()
        async with self._db.session() as session:
            drafts = (await session.execute(select(DefinitionRow))).scalars().all()
            versions = (await session.execute(select(VersionRow))).scalars().all()
        for row in drafts:
            if _references(load_definition(row.draft_json), resource_name):
                names.add(row.name)
        for version in versions:
            if _references(load_definition(version.definition_json), resource_name):
                names.add(version.name)
        return names

    @staticmethod
    def _redacted(resource: Resource) -> Resource:
        """Round-trip through the redacting serializer -> placeholders only."""
        try:
            return RESOURCE_ADAPTER.validate_python(
                RESOURCE_ADAPTER.dump_python(resource, mode="json")
            )
        except ValidationError:  # pragma: no cover - placeholder always validates
            return resource


def _references(defn: FlowDefinition, resource_name: str) -> bool:
    for node in defn.nodes:
        match node:
            case LlmCallNode() | RetrievalNode():
                if node.config.resource == resource_name:
                    return True
            case McpToolNode():
                if node.config.resource == resource_name:
                    return True
            case _:
                pass
    return False


__all__ = [
    "ResourceConflictError",
    "ResourceNotFoundError",
    "ResourceService",
    "ResourceValidationError",
    "secret_ref",
]
