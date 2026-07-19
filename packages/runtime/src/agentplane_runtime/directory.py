"""Registry-backed agent directory: resolves orchestrator sub-agent references.

Read-side counterpart of the self-registration in ``registration.py`` — the
same public registry API through the SDK, never a service import (architecture
invariant 3). Only agents that are registered (and enabled) resolve, and the
resolved URL is the gateway URL by registry invariant — every agent-to-agent
hop stays authenticated, rate-limited, and traced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from a2a.types import AgentCard

from agentplane_runtime.settings import RuntimeSettings
from agentplane_sdk import (
    AgentplaneError,
    OidcClientCredentialsProvider,
    RegistryClient,
    TokenProvider,
)

_CACHE_TTL_S = 30.0


class AgentResolutionError(RuntimeError):
    """Registry not configured or unreachable while resolving an agent reference."""


class AgentNotFoundError(LookupError):
    """No enabled A2A agent with that card name is registered."""


@dataclass(frozen=True)
class ResolvedAgent:
    """The slice of a registry entry the orchestrator needs to call an agent."""

    name: str
    url: str
    description: str
    examples: tuple[str, ...] = ()
    status: str = "unknown"  # registry health status; validation warns on "unhealthy"


class AgentDirectory:
    """Resolves registry card names to callable A2A agents, with a small TTL cache.

    The cache keeps the per-tool-call lookup off the registry's hot path; its
    TTL bounds how long a redeployed agent's old URL may still be used.
    """

    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._auth = self._build_auth()
        self._cache: dict[str, tuple[float, ResolvedAgent]] = {}

    def _build_auth(self) -> str | TokenProvider | None:
        """Prefer client-credentials (auto-refreshed) over a static token."""
        s = self._settings
        if s.registry_client_id and s.registry_client_secret:
            return OidcClientCredentialsProvider(
                s.registry_oidc_issuer or s.oidc_issuer,
                s.registry_client_id,
                s.registry_client_secret,
            )
        return s.registry_token or None

    @property
    def enabled(self) -> bool:
        return bool(self._settings.registry_url)

    async def resolve(self, name: str) -> ResolvedAgent:
        """Look up an enabled A2A agent by card name.

        Raises ``AgentResolutionError`` when no registry is configured or it is
        unreachable, ``AgentNotFoundError`` when no enabled agent matches.
        """
        cached = self._cache.get(name)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        if not self.enabled:
            raise AgentResolutionError("no registry configured (AGENTPLANE_RUNTIME_REGISTRY_URL)")
        try:
            async with RegistryClient(self._settings.registry_url, self._auth) as client:
                page = await client.search(name, kind="agent", limit=200)
        except AgentplaneError as exc:
            raise AgentResolutionError(f"registry lookup failed: {exc}") from exc
        for entry in page.items:
            if entry.card_name != name or not entry.enabled:
                continue
            card = entry.card
            examples: tuple[str, ...] = ()
            if isinstance(card, AgentCard):
                examples = tuple(example for skill in card.skills for example in skill.examples)
            resolved = ResolvedAgent(
                name=name,
                url=entry.url,
                description=str(card.description),
                examples=examples,
                status=entry.status,
            )
            self._cache[name] = (time.monotonic() + _CACHE_TTL_S, resolved)
            return resolved
        raise AgentNotFoundError(name)


__all__ = ["AgentDirectory", "AgentNotFoundError", "AgentResolutionError", "ResolvedAgent"]
