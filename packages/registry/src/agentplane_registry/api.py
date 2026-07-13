"""Registry REST API (SPEC §5.1), prefix ``/api/v1``."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from agentplane_core import (
    Capabilities,
    EntryKind,
    HealthStatus,
    Page,
    RegistryEntry,
    RegistryEntryCreate,
    RegistryEntryPatch,
    SearchQuery,
    serialize_card,
)
from agentplane_registry.auth import Principal
from agentplane_registry.db import Database, EntryRow, row_to_entry
from agentplane_registry.search import RegistrySearch
from agentplane_registry.settings import REGISTRY_VERSION, RegistrySettings
from agentplane_registry.urlcheck import is_private_url


@dataclass
class RegistryState:
    """Shared service state stored on the FastAPI app."""

    db: Database
    settings: RegistrySettings
    search: RegistrySearch


def _state(request: Request) -> RegistryState:
    state: RegistryState = request.app.state.registry
    return state


async def _principal(request: Request) -> Principal:
    principal: Principal = await request.app.state.authenticator.authenticate(request)
    return principal


State = Annotated[RegistryState, Depends(_state)]
Caller = Annotated[Principal, Depends(_principal)]

router = APIRouter(prefix="/api/v1")
health_router = APIRouter()


def _visible(row: EntryRow, caller: Principal, auth_mode: str) -> bool:
    if auth_mode == "none" or caller.is_admin:
        return True
    return row.owner == caller.sub


def _check_url(url: str, settings: RegistrySettings) -> None:
    if not settings.allow_private_urls and is_private_url(url):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="url must be a public gateway URL (set ALLOW_PRIVATE_URLS to override)",
        )


@router.post("/agents", status_code=status.HTTP_201_CREATED, response_model=RegistryEntry)
async def register_entry(body: RegistryEntryCreate, state: State, caller: Caller) -> RegistryEntry:
    _check_url(body.url, state.settings)
    now = datetime.now(UTC)
    row = EntryRow(
        id=str(uuid.uuid4()),
        kind=body.kind,
        name=body.card.name,
        owner=caller.sub,
        url=body.url,
        card_json=_dump_card(body),
        tags_json=list(body.tags),
        status="starting",
        created_at=now,
        updated_at=now,
    )
    try:
        async with state.db.session() as session, session.begin():
            session.add(row)
    except IntegrityError:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"entry named {body.card.name!r} already exists for this owner",
        ) from None
    entry = row_to_entry(row)
    await state.search.index(entry)
    return entry


def _dump_card(body: RegistryEntryCreate | RegistryEntryPatch) -> str:
    return json.dumps(serialize_card(body.card), sort_keys=True)


@router.get("/agents", response_model=Page)
async def list_entries(
    state: State,
    caller: Caller,
    kind: Annotated[EntryKind | None, Query()] = None,
    status_filter: Annotated[HealthStatus | None, Query(alias="status")] = None,
    tags: Annotated[str, Query(description="comma-separated, AND semantics")] = "",
    owner: Annotated[Literal["me", "all"], Query()] = "me",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page:
    if owner == "all" and state.settings.auth_mode == "oidc" and not caller.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="owner=all requires admin")
    stmt = select(EntryRow).order_by(EntryRow.created_at.desc())
    count_stmt = select(func.count()).select_from(EntryRow)
    if state.settings.auth_mode == "oidc" and owner == "me" and not caller.is_admin:
        stmt = stmt.where(EntryRow.owner == caller.sub)
        count_stmt = count_stmt.where(EntryRow.owner == caller.sub)
    if kind is not None:
        stmt = stmt.where(EntryRow.kind == kind)
        count_stmt = count_stmt.where(EntryRow.kind == kind)
    if status_filter is not None:
        stmt = stmt.where(EntryRow.status == status_filter)
        count_stmt = count_stmt.where(EntryRow.status == status_filter)

    async with state.db.session() as session:
        rows = (await session.execute(stmt)).scalars().all()

    wanted_tags = {t.strip() for t in tags.split(",") if t.strip()}
    entries = [
        row_to_entry(row) for row in rows if not wanted_tags or wanted_tags <= set(row.tags_json)
    ]
    total = len(entries)
    return Page(items=entries[offset : offset + limit], total=total, limit=limit, offset=offset)


@router.get("/agents/search", response_model=Page)
async def search_entries(
    state: State,
    caller: Caller,
    response: Response,
    q: Annotated[str, Query()] = "",
    tags: Annotated[str, Query()] = "",
    kind: Annotated[EntryKind | None, Query()] = None,
    status_filter: Annotated[HealthStatus | None, Query(alias="status")] = None,
    semantic: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page:
    owner_filter: str | None = None
    if state.settings.auth_mode == "oidc" and not caller.is_admin:
        owner_filter = caller.sub
    if semantic and not state.search.semantic_enabled:
        response.headers["X-Degraded"] = "semantic"
        semantic = False
    query = SearchQuery(
        q=q,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        kind=kind,
        status=status_filter,
        semantic=semantic,
        owner=owner_filter,
        limit=limit,
        offset=offset,
    )
    return await state.search.search(query)


@router.get("/agents/{entry_id}", response_model=RegistryEntry)
async def get_entry(entry_id: uuid.UUID, state: State, caller: Caller) -> RegistryEntry:
    row = await _load_visible(entry_id, state, caller)
    return row_to_entry(row)


async def _load_visible(entry_id: uuid.UUID, state: RegistryState, caller: Principal) -> EntryRow:
    async with state.db.session() as session:
        row = await session.get(EntryRow, str(entry_id))
    if row is None or not _visible(row, caller, state.settings.auth_mode):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="entry not found")
    return row


def _require_owner(row: EntryRow, caller: Principal, auth_mode: str) -> None:
    if auth_mode == "oidc" and not caller.is_admin and row.owner != caller.sub:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="owner or admin required")


@router.put("/agents/{entry_id}", response_model=RegistryEntry)
async def update_entry(
    entry_id: uuid.UUID, body: RegistryEntryPatch, state: State, caller: Caller
) -> RegistryEntry:
    row = await _load_visible(entry_id, state, caller)
    _require_owner(row, caller, state.settings.auth_mode)
    if body.url is not None:
        _check_url(body.url, state.settings)
    async with state.db.session() as session, session.begin():
        fresh = await session.get(EntryRow, str(entry_id))
        if fresh is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="entry not found")
        if body.card is not None:
            fresh.card_json = _dump_card(body)
            fresh.name = body.card.name
        if body.url is not None:
            fresh.url = body.url
        if body.tags is not None:
            fresh.tags_json = list(body.tags)
        fresh.updated_at = datetime.now(UTC)
        row = fresh
    entry = row_to_entry(row)
    await state.search.index(entry)
    return entry


@router.delete("/agents/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_entry(entry_id: uuid.UUID, state: State, caller: Caller) -> None:
    row = await _load_visible(entry_id, state, caller)
    _require_owner(row, caller, state.settings.auth_mode)
    async with state.db.session() as session, session.begin():
        fresh = await session.get(EntryRow, str(entry_id))
        if fresh is not None:
            await session.delete(fresh)
    await state.search.remove(entry_id)


@router.get("/capabilities", response_model=Capabilities)
async def capabilities(state: State) -> Capabilities:
    return Capabilities(
        semantic_search=state.search.semantic_enabled,
        auth=state.settings.auth_mode,
        version=REGISTRY_VERSION,
    )


@health_router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@health_router.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    state: RegistryState = request.app.state.registry
    async with state.db.session() as session:
        await session.execute(select(1))
    return {"status": "ready"}


__all__ = ["RegistryState", "health_router", "router"]
