"""Typed async clients for the registry and runtime APIs (SPEC §4.1).

Everything the SDK does is possible with plain HTTP — the SDK is
convenience, not a requirement.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from types import TracebackType
from typing import Any, Self, TypeVar
from uuid import UUID

import httpx
from pydantic import TypeAdapter

from agentplane_core import (
    Capabilities,
    DefinitionInfo,
    DeploymentInfo,
    FlowDefinition,
    Page,
    RegistryEntry,
    RegistryEntryCreate,
    RegistryEntryPatch,
    Resource,
    ValidationResult,
)
from agentplane_sdk.auth import TokenProvider, as_token_provider
from agentplane_sdk.errors import (
    ApiError,
    AuthError,
    ConflictError,
    NotFoundError,
    TransportError,
    ValidationFailedError,
)

_RESOURCE_ADAPTER: TypeAdapter[Resource] = TypeAdapter(Resource)
_RESOURCE_LIST_ADAPTER: TypeAdapter[list[Resource]] = TypeAdapter(list[Resource])
_DEFINITION_LIST_ADAPTER: TypeAdapter[list[DefinitionInfo]] = TypeAdapter(list[DefinitionInfo])

T = TypeVar("T")

# `list` is also a client method name (SPEC API); alias the builtin for annotations.
_List = list


class _BaseClient:
    """Shared HTTP plumbing: base URL, bearer auth, error mapping."""

    def __init__(
        self,
        base_url: str,
        token: str | TokenProvider | None = None,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token_provider = as_token_provider(token)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        params: Mapping[str, str | int | None] | None = None,
    ) -> httpx.Response:
        headers: dict[str, str] = {}
        if self._token_provider is not None:
            headers["Authorization"] = f"Bearer {await self._token_provider.get_token()}"
        try:
            response = await self._client.request(
                method,
                path,
                json=json,
                params=self._clean_params(params),
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise TransportError(str(exc)) from exc
        self._raise_for_status(response)
        return response

    @staticmethod
    def _clean_params(
        params: Mapping[str, str | int | None] | None,
    ) -> dict[str, str | int] | None:
        if params is None:
            return None
        return {k: v for k, v in params.items() if v is not None}

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status < httpx.codes.BAD_REQUEST:
            return
        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise AuthError(f"HTTP {status}: {response.text}")
        if status == httpx.codes.NOT_FOUND:
            raise NotFoundError(response.text)
        if status == httpx.codes.CONFLICT:
            raise ConflictError(response.text)
        if status == httpx.codes.UNPROCESSABLE_ENTITY:
            try:
                result = ValidationResult.model_validate(response.json())
            except ValueError:
                raise ApiError(status, response.text) from None
            raise ValidationFailedError(result)
        raise ApiError(status, response.text)


class RegistryClient(_BaseClient):
    """Client for the agentplane-registry REST API (SPEC §5.1)."""

    _prefix = "/api/v1"

    async def register(self, entry: RegistryEntryCreate) -> RegistryEntry:
        response = await self._request(
            "POST", f"{self._prefix}/agents", json=entry.model_dump(mode="json")
        )
        return RegistryEntry.model_validate(response.json())

    async def get(self, id: UUID) -> RegistryEntry:
        response = await self._request("GET", f"{self._prefix}/agents/{id}")
        return RegistryEntry.model_validate(response.json())

    async def update(self, id: UUID, patch: RegistryEntryPatch) -> RegistryEntry:
        response = await self._request(
            "PUT",
            f"{self._prefix}/agents/{id}",
            json=patch.model_dump(mode="json", exclude_none=True),
        )
        return RegistryEntry.model_validate(response.json())

    async def delete(self, id: UUID) -> None:
        await self._request("DELETE", f"{self._prefix}/agents/{id}")

    async def search(
        self,
        q: str = "",
        tags: _List[str] | None = None,
        kind: str | None = None,
        status: str | None = None,
        semantic: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        params: dict[str, str | int | None] = {
            "q": q,
            "kind": kind,
            "status": status,
            "semantic": "true" if semantic else None,
            "limit": limit,
            "offset": offset,
        }
        if tags:
            params["tags"] = ",".join(tags)
        response = await self._request("GET", f"{self._prefix}/agents/search", params=params)
        return Page.model_validate(response.json())

    async def list(
        self,
        kind: str | None = None,
        status: str | None = None,
        tags: _List[str] | None = None,
        owner: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        params: dict[str, str | int | None] = {
            "kind": kind,
            "status": status,
            "owner": owner,
            "limit": limit,
            "offset": offset,
        }
        if tags:
            params["tags"] = ",".join(tags)
        response = await self._request("GET", f"{self._prefix}/agents", params=params)
        return Page.model_validate(response.json())

    async def capabilities(self) -> Capabilities:
        response = await self._request("GET", f"{self._prefix}/capabilities")
        return Capabilities.model_validate(response.json())


class RuntimeClient(_BaseClient):
    """Client for the agentplane-runtime definitions/resources API (SPEC §6.1/§6.3)."""

    _prefix = "/api/v1"

    async def validate(self, defn: FlowDefinition | Mapping[str, object]) -> ValidationResult:
        payload = defn.canonical_dict() if isinstance(defn, FlowDefinition) else dict(defn)
        response = await self._request("POST", f"{self._prefix}/definitions/validate", json=payload)
        return ValidationResult.model_validate(response.json())

    async def create_draft(self, defn: FlowDefinition) -> DefinitionInfo:
        response = await self._request(
            "POST", f"{self._prefix}/definitions", json=defn.canonical_dict()
        )
        return DefinitionInfo.model_validate(response.json())

    async def update_draft(self, name: str, defn: FlowDefinition) -> DefinitionInfo:
        response = await self._request(
            "PUT", f"{self._prefix}/definitions/{name}", json=defn.canonical_dict()
        )
        return DefinitionInfo.model_validate(response.json())

    async def deploy(
        self,
        name: str,
        *,
        version: int | None = None,
        version_label: str | None = None,
        ephemeral: bool = False,
    ) -> DeploymentInfo:
        """Deploy the draft (optionally labelled) or re-serve an existing version.

        ``version`` rolls back by deploy counter, ``version_label`` by semantic
        version — a label that already exists rolls back to it instead of
        freezing a new version.
        """
        params: dict[str, str | int | None] = {
            "version": version,
            "version_label": version_label,
            "ephemeral": "true" if ephemeral else None,
        }
        response = await self._request(
            "POST", f"{self._prefix}/definitions/{name}/deploy", params=params
        )
        return DeploymentInfo.model_validate(response.json())

    async def undeploy(self, name: str) -> None:
        await self._request("POST", f"{self._prefix}/definitions/{name}/undeploy")

    async def get(self, name: str, *, include_definition: bool = False) -> DefinitionInfo:
        params = {"include": "definition"} if include_definition else None
        response = await self._request("GET", f"{self._prefix}/definitions/{name}", params=params)
        return DefinitionInfo.model_validate(response.json())

    async def list(self, status: str | None = None) -> _List[DefinitionInfo]:
        response = await self._request(
            "GET", f"{self._prefix}/definitions", params={"status": status}
        )
        return _DEFINITION_LIST_ADAPTER.validate_python(response.json())

    async def export(self, name: str, version: int | None = None) -> FlowDefinition:
        response = await self._request(
            "GET", f"{self._prefix}/definitions/{name}/export", params={"version": version}
        )
        return FlowDefinition.model_validate(response.json())

    async def delete(self, name: str) -> None:
        await self._request("DELETE", f"{self._prefix}/definitions/{name}")

    async def create_resource(self, resource: Resource) -> Resource:
        response = await self._request(
            "POST",
            f"{self._prefix}/resources",
            json=_RESOURCE_ADAPTER.dump_python(
                resource, mode="json", context={"reveal_secrets": True}
            ),
        )
        return _RESOURCE_ADAPTER.validate_python(response.json())

    async def list_resources(self, kind: str | None = None) -> _List[Resource]:
        response = await self._request("GET", f"{self._prefix}/resources", params={"kind": kind})
        return _RESOURCE_LIST_ADAPTER.validate_python(response.json())

    async def delete_resource(self, name: str) -> None:
        await self._request("DELETE", f"{self._prefix}/resources/{name}")


def _run_sync[R](coro: Coroutine[Any, Any, R]) -> R:
    return asyncio.run(coro)


class SyncRuntimeClient:
    """Blocking convenience wrapper generated via ``asyncio.run`` (SPEC §4.1)."""

    def __init__(self, base_url: str, token: str | TokenProvider | None = None) -> None:
        self._base_url = base_url
        self._token = token

    def _call(self, fn: Callable[[RuntimeClient], Coroutine[Any, Any, T]]) -> T:
        async def runner() -> T:
            async with RuntimeClient(self._base_url, self._token) as client:
                return await fn(client)

        return _run_sync(runner())

    def validate(self, defn: FlowDefinition | Mapping[str, object]) -> ValidationResult:
        return self._call(lambda c: c.validate(defn))

    def create_draft(self, defn: FlowDefinition) -> DefinitionInfo:
        return self._call(lambda c: c.create_draft(defn))

    def update_draft(self, name: str, defn: FlowDefinition) -> DefinitionInfo:
        return self._call(lambda c: c.update_draft(name, defn))

    def deploy(
        self,
        name: str,
        *,
        version: int | None = None,
        version_label: str | None = None,
        ephemeral: bool = False,
    ) -> DeploymentInfo:
        return self._call(
            lambda c: c.deploy(
                name, version=version, version_label=version_label, ephemeral=ephemeral
            )
        )

    def undeploy(self, name: str) -> None:
        return self._call(lambda c: c.undeploy(name))

    def get(self, name: str, *, include_definition: bool = False) -> DefinitionInfo:
        return self._call(lambda c: c.get(name, include_definition=include_definition))

    def list(self, status: str | None = None) -> _List[DefinitionInfo]:
        return self._call(lambda c: c.list(status))

    def export(self, name: str, version: int | None = None) -> FlowDefinition:
        return self._call(lambda c: c.export(name, version))

    def delete(self, name: str) -> None:
        return self._call(lambda c: c.delete(name))

    def create_resource(self, resource: Resource) -> Resource:
        return self._call(lambda c: c.create_resource(resource))

    def list_resources(self, kind: str | None = None) -> _List[Resource]:
        return self._call(lambda c: c.list_resources(kind))

    def delete_resource(self, name: str) -> None:
        return self._call(lambda c: c.delete_resource(name))


class SyncRegistryClient:
    """Blocking convenience wrapper generated via ``asyncio.run`` (SPEC §4.1)."""

    def __init__(self, base_url: str, token: str | TokenProvider | None = None) -> None:
        self._base_url = base_url
        self._token = token

    def _call(self, fn: Callable[[RegistryClient], Coroutine[Any, Any, T]]) -> T:
        async def runner() -> T:
            async with RegistryClient(self._base_url, self._token) as client:
                return await fn(client)

        return _run_sync(runner())

    def register(self, entry: RegistryEntryCreate) -> RegistryEntry:
        return self._call(lambda c: c.register(entry))

    def get(self, id: UUID) -> RegistryEntry:
        return self._call(lambda c: c.get(id))

    def update(self, id: UUID, patch: RegistryEntryPatch) -> RegistryEntry:
        return self._call(lambda c: c.update(id, patch))

    def delete(self, id: UUID) -> None:
        return self._call(lambda c: c.delete(id))

    def search(
        self,
        q: str = "",
        tags: _List[str] | None = None,
        kind: str | None = None,
        status: str | None = None,
        semantic: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        return self._call(
            lambda c: c.search(
                q,
                tags=tags,
                kind=kind,
                status=status,
                semantic=semantic,
                limit=limit,
                offset=offset,
            )
        )

    def capabilities(self) -> Capabilities:
        return self._call(lambda c: c.capabilities())


__all__ = [
    "RegistryClient",
    "RuntimeClient",
    "SyncRegistryClient",
    "SyncRuntimeClient",
]
