"""Runtime REST API (SPEC §6.1/§6.3), prefix ``/api/v1``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from agentplane_core import (
    DefinitionInfo,
    DeploymentInfo,
    FlowDefinition,
    JsonObject,
    Resource,
    ValidationResult,
    VersionLabel,
)
from agentplane_runtime.auth import AccessScope, Principal
from agentplane_runtime.definitions import (
    DefinitionConflictError,
    DefinitionInvalidError,
    DefinitionNotFoundError,
    DefinitionQuotaError,
    DefinitionService,
    DefinitionStateError,
)
from agentplane_runtime.resources import (
    ResourceConflictError,
    ResourceNotFoundError,
    ResourceService,
    ResourceValidationError,
)


@dataclass
class RuntimeState:
    definitions: DefinitionService
    resources: ResourceService
    auth_mode: str = "none"
    builder_role: str = "builder"


def _state(request: Request) -> RuntimeState:
    state: RuntimeState = request.app.state.runtime
    return state


async def _principal(request: Request) -> Principal:
    principal: Principal = await request.app.state.authenticator.authenticate(request)
    return principal


State = Annotated[RuntimeState, Depends(_state)]
Caller = Annotated[Principal, Depends(_principal)]


def _scope(state: RuntimeState, caller: Principal) -> AccessScope:
    """What the caller may see/manage: their own rows plus their teams' (SPEC §7.1)."""
    return AccessScope.for_caller(caller, state.auth_mode)


def _require_builder(state: RuntimeState, caller: Principal) -> None:
    """Writes need the builder role or admin under OIDC (SPEC §7.1); reads stay role-free.

    A missing role is not an existence question, so this is a 403 — foreign-object
    404 semantics (ownership scope) are unchanged for callers who do hold the role.
    """
    if state.auth_mode != "oidc" or caller.is_admin or state.builder_role in caller.roles:
        return
    raise HTTPException(
        status.HTTP_403_FORBIDDEN,
        detail=f"write operations require the {state.builder_role!r} role",
    )


def _chosen_group(state: RuntimeState, caller: Principal, group: str) -> str:
    """Validate the group a caller assigns at create time.

    Off/admin may set any group; a regular user only one of their own. An empty
    group means "private" (owner-only).
    """
    if not group or state.auth_mode != "oidc" or caller.is_admin:
        return group
    if group not in caller.groups:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=f"not a member of group {group!r}")
    return group


router = APIRouter(prefix="/api/v1")
health_router = APIRouter()


def _validation_response(result: ValidationResult) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=result.model_dump(mode="json"),
    )


@router.post("/definitions/validate", response_model=ValidationResult)
async def validate_definition(body: JsonObject, state: State, caller: Caller) -> ValidationResult:
    """Always 200 with a ValidationResult, even when invalid (SPEC §6.1)."""
    return await state.definitions.validate(dict(body))


@router.post("/definitions", status_code=status.HTTP_201_CREATED, response_model=DefinitionInfo)
async def create_definition(
    body: FlowDefinition,
    state: State,
    caller: Caller,
    group: Annotated[str, Query(description="team that owns the flow; empty = private")] = "",
) -> DefinitionInfo | JSONResponse:
    _require_builder(state, caller)
    try:
        return await state.definitions.create_draft(
            body, caller.sub, group=_chosen_group(state, caller, group), owner_name=caller.username
        )
    except DefinitionInvalidError as exc:
        return _validation_response(exc.result)
    except DefinitionConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None


@router.put("/definitions/{name}", response_model=DefinitionInfo)
async def update_definition(
    name: str, body: FlowDefinition, state: State, caller: Caller
) -> DefinitionInfo | JSONResponse:
    _require_builder(state, caller)
    try:
        return await state.definitions.update_draft(name, body, scope=_scope(state, caller))
    except DefinitionInvalidError as exc:
        return _validation_response(exc.result)
    except DefinitionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no definition {name!r}") from None
    except DefinitionConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None


@router.post("/definitions/{name}/deploy", response_model=DeploymentInfo)
async def deploy_definition(
    name: str,
    state: State,
    caller: Caller,
    version: Annotated[int | None, Query(ge=1)] = None,
    version_label: Annotated[VersionLabel | None, Query()] = None,
    ephemeral: Annotated[bool, Query()] = False,
) -> DeploymentInfo | JSONResponse:
    _require_builder(state, caller)  # ephemeral playground deploys count as writes
    try:
        return await state.definitions.deploy(
            name,
            version=version,
            version_label=version_label,
            ephemeral=ephemeral,
            bypass_quota=caller.is_admin,
            scope=_scope(state, caller),
        )
    except DefinitionInvalidError as exc:
        return _validation_response(exc.result)
    except DefinitionQuotaError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, detail=exc.body.model_dump(mode="json")
        ) from None
    except DefinitionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from None
    except DefinitionConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None


@router.post("/definitions/{name}/undeploy", status_code=status.HTTP_204_NO_CONTENT)
async def undeploy_definition(name: str, state: State, caller: Caller) -> None:
    _require_builder(state, caller)
    try:
        await state.definitions.undeploy(name, scope=_scope(state, caller))
    except DefinitionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no definition {name!r}") from None
    except DefinitionStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None


@router.get("/definitions", response_model=list[DefinitionInfo])
async def list_definitions(
    state: State,
    caller: Caller,
    status_filter: Annotated[
        Literal["draft", "deployed", "undeployed"] | None, Query(alias="status")
    ] = None,
) -> list[DefinitionInfo]:
    return await state.definitions.list(status_filter, scope=_scope(state, caller))


@router.get("/definitions/{name}", response_model=DefinitionInfo)
async def get_definition(
    name: str,
    state: State,
    caller: Caller,
    include: Annotated[Literal["definition"] | None, Query()] = None,
) -> DefinitionInfo:
    try:
        return await state.definitions.info(
            name, include_definition=include == "definition", scope=_scope(state, caller)
        )
    except DefinitionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no definition {name!r}") from None


@router.get("/definitions/{name}/export")
async def export_definition(
    name: str,
    state: State,
    caller: Caller,
    version: Annotated[int | None, Query(ge=1)] = None,
) -> JsonObject:
    try:
        defn = await state.definitions.export(name, version, scope=_scope(state, caller))
    except DefinitionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from None
    return defn.canonical_dict()


@router.delete("/definitions/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_definition(name: str, state: State, caller: Caller) -> None:
    _require_builder(state, caller)
    try:
        await state.definitions.delete(name, scope=_scope(state, caller))
    except DefinitionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no definition {name!r}") from None
    except DefinitionStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None


@router.post("/resources", status_code=status.HTTP_201_CREATED, response_model=Resource)
async def create_resource(
    body: Resource,
    state: State,
    caller: Caller,
    group: Annotated[str, Query(description="team that owns the resource; empty = private")] = "",
) -> Resource:
    _require_builder(state, caller)
    try:
        return await state.resources.create(
            body, caller.sub, group=_chosen_group(state, caller, group)
        )
    except ResourceConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None
    except ResourceValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.result.model_dump(mode="json")
        ) from None


@router.get("/resources", response_model=list[Resource])
async def list_resources(
    state: State,
    caller: Caller,
    kind: Annotated[str | None, Query()] = None,
) -> list[Resource]:
    return await state.resources.list(kind, scope=_scope(state, caller))


@router.get("/resources/{name}", response_model=Resource)
async def get_resource(name: str, state: State, caller: Caller) -> Resource:
    try:
        return await state.resources.get(name, scope=_scope(state, caller))
    except ResourceNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no resource {name!r}") from None


@router.put("/resources/{name}", response_model=Resource)
async def update_resource(name: str, body: Resource, state: State, caller: Caller) -> Resource:
    _require_builder(state, caller)
    try:
        return await state.resources.update(name, body, scope=_scope(state, caller))
    except ResourceNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no resource {name!r}") from None
    except ResourceValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.result.model_dump(mode="json")
        ) from None


@router.delete("/resources/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_resource(name: str, state: State, caller: Caller) -> None:
    _require_builder(state, caller)
    try:
        await state.resources.delete(name, scope=_scope(state, caller))
    except ResourceNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no resource {name!r}") from None
    except ResourceConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None


@health_router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@health_router.get("/readyz")
async def readyz() -> dict[str, str]:
    return {"status": "ready"}


__all__ = ["RuntimeState", "health_router", "router"]
