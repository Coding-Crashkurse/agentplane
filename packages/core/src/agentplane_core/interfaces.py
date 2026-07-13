"""Abstract interfaces implemented by the services (SPEC §3.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agentplane_core.registry import Page, RegistryEntry, SearchQuery
from agentplane_core.types import JsonObject


class ScoredHit(BaseModel):
    """One vector search hit."""

    model_config = ConfigDict(frozen=True)

    key: str
    score: float
    payload: JsonObject = Field(default_factory=dict)


class SecretsProvider(ABC):
    """Stores credentials referenced by name; values never leave the provider.

    Default implementation in the runtime: Fernet-encrypted DB column.
    """

    @abstractmethod
    async def put(self, ref: str, value: str) -> None: ...

    @abstractmethod
    async def get(self, ref: str) -> str: ...

    @abstractmethod
    async def delete(self, ref: str) -> None: ...


class VectorStore(ABC):
    """Vector index used by registry ``[semantic]`` and runtime retrieval."""

    @abstractmethod
    async def upsert(self, key: str, vector: list[float], payload: JsonObject) -> None: ...

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter: Mapping[str, JsonValue] | None = None,
    ) -> list[ScoredHit]: ...


class SearchBackend(ABC):
    """Registry search abstraction."""

    @abstractmethod
    async def index(self, entry: RegistryEntry) -> None: ...

    @abstractmethod
    async def remove(self, entry_id: UUID) -> None: ...

    @abstractmethod
    async def search(self, q: SearchQuery) -> Page: ...


__all__ = ["ScoredHit", "SearchBackend", "SecretsProvider", "VectorStore"]
