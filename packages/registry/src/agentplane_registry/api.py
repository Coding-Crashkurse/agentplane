"""Registry REST API (SPEC §5.1), prefix ``/api/v1``."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import delete, func, or_, select
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
    StatusEvent,
    StatusHistory,
    serialize_card,
)
from agentplane_registry.auth import AccessScope, Principal
from agentplane_registry.db import (
    Database,
    EntryRow,
    EntryStatusEventRow,
    record_status_event,
    row_to_entry,
)
from agentplane_registry.health import HealthJob
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


def _health_job(request: Request) -> HealthJob | None:
    job: HealthJob | None = getattr(request.app.state, "health_job", None)
    return job


State = Annotated[RegistryState, Depends(_state)]
Caller = Annotated[Principal, Depends(_principal)]
Health = Annotated[HealthJob | None, Depends(_health_job)]

router = APIRouter(prefix="/api/v1")
health_router = APIRouter()


def _visible(row: EntryRow, caller: Principal, auth_mode: str) -> bool:
    return AccessScope.for_caller(caller, auth_mode).allows(row.owner, row.group)


def _attribution(
    body: RegistryEntryCreate, caller: Principal, auth_mode: str
) -> tuple[str, str, str]:
    """The (owner, group, owner_name) to record.

    A trusted admin caller (the runtime publishing on behalf of a user) may
    assert all three. A regular caller owns what they register and may only
    attribute it to one of their own groups; the display name always comes
    from their own token then.
    """
    if caller.is_admin and auth_mode == "oidc":
        owner = body.owner or caller.sub
        # An asserted owner needs an asserted name; for self-registration the
        # admin's own token name is the fallback.
        owner_name = body.owner_name or ("" if body.owner else caller.username)
        return owner, body.group or "", owner_name
    group = body.group or ""
    if group and group not in caller.groups:
        group = ""
    return caller.sub, group, caller.username


def _check_url(url: str, settings: RegistrySettings) -> None:
    if not settings.allow_private_urls and is_private_url(url):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="url must be a public gateway URL (set ALLOW_PRIVATE_URLS to override)",
        )


@router.post("/agents", status_code=status.HTTP_201_CREATED, response_model=RegistryEntry)
async def register_entry(
    body: RegistryEntryCreate, state: State, caller: Caller, health: Health
) -> RegistryEntry:
    _check_url(body.url, state.settings)
    now = datetime.now(UTC)
    owner, group, owner_name = _attribution(body, caller, state.settings.auth_mode)
    row = EntryRow(
        id=str(uuid.uuid4()),
        kind=body.kind,
        name=body.card.name,
        owner=owner,
        owner_name=owner_name,
        group=group,
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
            record_status_event(session, row.id, "starting")
    except IntegrityError:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"entry named {body.card.name!r} already exists for this owner",
        ) from None
    entry = row_to_entry(row)
    if health is not None:
        # Check the fresh entry immediately instead of waiting for the next pass.
        health.kick(row.id)
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
    enabled: Annotated[
        bool | None,
        Query(description="Filter by enabled state; the management list shows both by default."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page:
    if owner == "all" and state.settings.auth_mode == "oidc" and not caller.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="owner=all requires admin")
    stmt = select(EntryRow).order_by(EntryRow.created_at.desc())
    count_stmt = select(func.count()).select_from(EntryRow)
    scope = AccessScope.for_caller(caller, state.settings.auth_mode)
    if owner == "me" and not scope.unrestricted:
        conditions = [EntryRow.owner == scope.sub]
        if scope.groups:
            conditions.append(EntryRow.group.in_(scope.groups))
        stmt = stmt.where(or_(*conditions))
        count_stmt = count_stmt.where(or_(*conditions))
    if kind is not None:
        stmt = stmt.where(EntryRow.kind == kind)
        count_stmt = count_stmt.where(EntryRow.kind == kind)
    if status_filter is not None:
        stmt = stmt.where(EntryRow.status == status_filter)
        count_stmt = count_stmt.where(EntryRow.status == status_filter)
    if enabled is not None:
        stmt = stmt.where(EntryRow.enabled.is_(enabled))
        count_stmt = count_stmt.where(EntryRow.enabled.is_(enabled))

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
    include_disabled: Annotated[
        bool, Query(description="Search is discovery: disabled entries are excluded by default.")
    ] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page:
    scope = AccessScope.for_caller(caller, state.settings.auth_mode)
    if semantic and not state.search.semantic_enabled:
        response.headers["X-Degraded"] = "semantic"
        semantic = False
    query = SearchQuery(
        q=q,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        kind=kind,
        status=status_filter,
        semantic=semantic,
        include_disabled=include_disabled,
        owner=None if scope.unrestricted else scope.sub,
        groups=[] if scope.unrestricted else sorted(scope.groups),
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
    """Mutations require the owner, a member of the entry's group, or an admin."""
    if not AccessScope.for_caller(caller, auth_mode).allows(row.owner, row.group):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail="owner, group member or admin required"
        )


def _require_owner_or_admin(row: EntryRow, caller: Principal, auth_mode: str) -> None:
    """Deletion is destructive: group members may edit, but not remove (SPEC §5.1)."""
    scope = AccessScope.for_caller(caller, auth_mode)
    if scope.unrestricted or row.owner == scope.sub:
        return
    raise HTTPException(status.HTTP_403_FORBIDDEN, detail="owner or admin required")


@router.put("/agents/{entry_id}", response_model=RegistryEntry)
async def update_entry(
    entry_id: uuid.UUID, body: RegistryEntryPatch, state: State, caller: Caller, health: Health
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
        if body.enabled is not None and body.enabled != fresh.enabled:
            fresh.enabled = body.enabled
            # Disabled entries are not monitored; re-enabling re-enters the
            # fast "starting" recheck so the entry turns healthy quickly.
            fresh.status = "starting" if body.enabled else "unknown"
            record_status_event(session, fresh.id, fresh.status)
        fresh.updated_at = datetime.now(UTC)
        row = fresh
    entry = row_to_entry(row)
    if body.enabled is True and entry.status == "starting" and health is not None:
        # Re-enabled: check immediately instead of waiting for the next pass.
        health.kick(str(entry_id))
    await state.search.index(entry)
    return entry


@router.delete("/agents/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_entry(entry_id: uuid.UUID, state: State, caller: Caller) -> None:
    row = await _load_visible(entry_id, state, caller)
    _require_owner_or_admin(row, caller, state.settings.auth_mode)
    async with state.db.session() as session, session.begin():
        fresh = await session.get(EntryRow, str(entry_id))
        if fresh is not None:
            await session.delete(fresh)
        await session.execute(
            delete(EntryStatusEventRow).where(EntryStatusEventRow.entry_id == str(entry_id))
        )
    await state.search.remove(entry_id)


@router.get("/agents/{entry_id}/history", response_model=StatusHistory)
async def entry_history(
    entry_id: uuid.UUID,
    state: State,
    caller: Caller,
    hours: Annotated[float, Query(gt=0, le=24 * 90)] = 24.0,
) -> StatusHistory:
    """Status transitions within the window, plus the one in effect at its start."""
    await _load_visible(entry_id, state, caller)
    window_start = datetime.now(UTC) - timedelta(hours=hours)
    in_window = (
        select(EntryStatusEventRow)
        .where(EntryStatusEventRow.entry_id == str(entry_id))
        .where(EntryStatusEventRow.at >= window_start)
        .order_by(EntryStatusEventRow.at)
    )
    before_window = (
        select(EntryStatusEventRow)
        .where(EntryStatusEventRow.entry_id == str(entry_id))
        .where(EntryStatusEventRow.at < window_start)
        .order_by(EntryStatusEventRow.at.desc())
        .limit(1)
    )
    async with state.db.session() as session:
        events = list((await session.execute(in_window)).scalars().all())
        prior = (await session.execute(before_window)).scalar_one_or_none()
    if prior is not None:
        events.insert(0, prior)
    return StatusHistory(
        items=[StatusEvent.model_validate({"status": e.status, "at": e.at}) for e in events],
        window_h=hours,
        retention_h=state.settings.history_retention_h,
    )


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
