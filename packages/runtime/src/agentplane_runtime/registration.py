"""Self-registration with the registry (SPEC §6.5).

Registry unreachable -> log + retry with backoff; serving is NOT blocked by
registry availability.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any
from uuid import UUID

from agentplane_core import (
    FlowDefinition,
    RegistryEntryCreate,
    RegistryEntryPatch,
    ToolCard,
    ToolCardTool,
)
from agentplane_core.registry import Card
from agentplane_runtime.serving import build_agent_card
from agentplane_runtime.settings import RuntimeSettings
from agentplane_sdk import AgentplaneError, ConflictError, RegistryClient

logger = logging.getLogger(__name__)

_RETRY_DELAYS_S = (1.0, 5.0, 15.0, 60.0)


def build_card(defn: FlowDefinition, public_url: str) -> Card:
    if defn.expose.kind == "mcp":
        return ToolCard(
            name=defn.name,
            description=defn.description,
            url=public_url,
            tools=[
                ToolCardTool(
                    name=defn.expose.tool_name or defn.name.replace("-", "_"),
                    description=defn.expose.tool_description or defn.description,
                )
            ],
        )
    return build_agent_card(defn, public_url)


class RegistryRegistrar:
    """Registers/deregisters deployed flows, with background retry."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def enabled(self) -> bool:
        return bool(self._settings.registry_url)

    def _client(self) -> RegistryClient:
        token = self._settings.registry_token or None
        return RegistryClient(self._settings.registry_url, token)

    async def register(
        self,
        defn: FlowDefinition,
        public_url: str,
        existing_id: UUID | None,
        owner: str = "",
        group: str = "",
    ) -> UUID | None:
        """One immediate attempt; on failure, retries continue in the background.

        ``owner``/``group`` attribute the entry to the flow's author and team.
        The registry honors them only when this runtime authenticates as an admin
        service, so published entries are attributed to the flow author rather
        than the runtime (SPEC §7.1).
        """
        if not self.enabled:
            return None
        try:
            return await self._register_once(defn, public_url, existing_id, owner, group)
        except AgentplaneError as exc:
            logger.warning("registry registration failed (%s); retrying in background", exc)
            self._spawn(self._retry_register(defn, public_url, existing_id, owner, group))
            return existing_id

    async def _register_once(
        self,
        defn: FlowDefinition,
        public_url: str,
        existing_id: UUID | None,
        owner: str = "",
        group: str = "",
    ) -> UUID:
        card = build_card(defn, public_url)
        kind = "mcp_server" if defn.expose.kind == "mcp" else "agent"
        create = RegistryEntryCreate(
            kind=kind,
            card=card,
            url=public_url,
            tags=list(defn.tags),
            owner=owner or None,
            group=group or None,
        )
        async with self._client() as client:
            if existing_id is not None:
                try:
                    entry = await client.update(
                        existing_id,
                        RegistryEntryPatch(card=card, url=public_url, tags=list(defn.tags)),
                    )
                except AgentplaneError:
                    entry = await client.register(create)
                return entry.id
            try:
                entry = await client.register(create)
            except ConflictError:
                # entry exists from a previous run we lost track of; find + update
                page = await client.search(defn.name, kind=kind, limit=200)
                for candidate in page.items:
                    if candidate.card_name == defn.name:
                        return (
                            await client.update(
                                candidate.id,
                                RegistryEntryPatch(card=card, url=public_url, tags=list(defn.tags)),
                            )
                        ).id
                raise
            return entry.id

    async def _retry_register(
        self,
        defn: FlowDefinition,
        public_url: str,
        existing_id: UUID | None,
        owner: str = "",
        group: str = "",
    ) -> None:
        for delay in _RETRY_DELAYS_S:
            await asyncio.sleep(delay)
            try:
                await self._register_once(defn, public_url, existing_id, owner, group)
            except AgentplaneError as exc:
                logger.warning("registry retry failed (%s)", exc)
                continue
            return
        logger.error("giving up registering %s with the registry", defn.name)

    async def deregister(self, registry_id: UUID | None) -> None:
        if not self.enabled or registry_id is None:
            return
        try:
            async with self._client() as client:
                await client.delete(registry_id)
        except AgentplaneError as exc:
            logger.warning("registry deregistration failed (%s)", exc)

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()


__all__ = ["RegistryRegistrar", "build_card"]
