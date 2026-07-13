"""Resource models — named platform-level connections (SPEC §3.2).

Secret fields are write-only: accepted on create/update, never serialized
back (responses render ``"•••"``), stored via ``SecretsProvider``. Clients
that must transmit the real value (create/update requests) dump with
``context={"reveal_secrets": True}``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FieldSerializationInfo, field_serializer

from agentplane_core.types import Slug

SECRET_PLACEHOLDER = "•••"


def _redact(value: str | None, info: FieldSerializationInfo) -> str | None:
    if value is None:
        return None
    context = info.context
    if isinstance(context, dict) and context.get("reveal_secrets"):
        return value
    return SECRET_PLACEHOLDER


class EmbeddingConfig(BaseModel):
    """Embedding settings bound to a vector DB collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: Slug = Field(description="ModelProvider resource name used to embed queries")
    model: str
    dimension: int = Field(ge=1)


class _ResourceBase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Slug
    display_name: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ModelProviderResource(_ResourceBase):
    kind: Literal["model_provider"] = "model_provider"
    base_url: str = Field(
        default="",
        description="OpenAI-compatible base URL; empty means the gateway's LLM endpoint",
    )
    api_key_secret: str | None = Field(default=None, description="write-only")
    default_model: str = ""

    @field_serializer("api_key_secret")
    def _redact_api_key(self, value: str | None, info: FieldSerializationInfo) -> str | None:
        return _redact(value, info)


class VectorDBResource(_ResourceBase):
    """Existing vector database, consumed read-only by contract."""

    kind: Literal["pgvector", "qdrant"]
    url: str = ""
    dsn_secret: str | None = Field(default=None, description="write-only")
    api_key_secret: str | None = Field(default=None, description="write-only")
    embedding: EmbeddingConfig

    @field_serializer("dsn_secret", "api_key_secret")
    def _redact_secrets(self, value: str | None, info: FieldSerializationInfo) -> str | None:
        return _redact(value, info)


class McpServerResource(_ResourceBase):
    kind: Literal["mcp_server"] = "mcp_server"
    url: str = Field(description="gateway URL for platform tools; external URL allowed")
    auth_secret: str | None = Field(default=None, description="write-only optional bearer")

    @field_serializer("auth_secret")
    def _redact_auth(self, value: str | None, info: FieldSerializationInfo) -> str | None:
        return _redact(value, info)


Resource = Annotated[
    ModelProviderResource | VectorDBResource | McpServerResource,
    Field(discriminator="kind"),
]

ResourceKind = Literal["model_provider", "pgvector", "qdrant", "mcp_server"]

VECTOR_DB_KINDS: frozenset[str] = frozenset({"pgvector", "qdrant"})

__all__ = [
    "SECRET_PLACEHOLDER",
    "VECTOR_DB_KINDS",
    "EmbeddingConfig",
    "McpServerResource",
    "ModelProviderResource",
    "Resource",
    "ResourceKind",
    "VectorDBResource",
]
