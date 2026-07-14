"""Registry, validation and deployment contract models (SPEC §3.4).

The ``card`` of an agent entry is the official ``a2a.types.AgentCard``
protobuf message (never a proprietary re-definition); on the wire it is the
A2A JSON card format. MCP servers use the reduced ``ToolCard``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from a2a.types import AgentCard
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_serializer,
    model_validator,
)

from agentplane_core.card_json import agent_card_from_dict, agent_card_to_json_dict
from agentplane_core.definition import FlowDefinition
from agentplane_core.types import JsonObject, JsonSchema, Slug, VersionLabel

EntryKind = Literal["agent", "mcp_server"]
HealthStatus = Literal["starting", "healthy", "unhealthy", "unknown"]
AuthMode = Literal["none", "oidc"]


class ToolCardTool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    input_schema: JsonSchema | None = None


class ToolCard(BaseModel):
    """Reduced discovery document for MCP servers (SPEC §3.4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    url: str = ""
    tools: list[ToolCardTool] = Field(default_factory=list)


# In-memory the card is the official protobuf AgentCard (agents) or the
# reduced ToolCard (MCP servers); on the wire both are plain JSON objects.
Card = Annotated[
    AgentCard | ToolCard,
    WithJsonSchema(
        {
            "type": "object",
            "description": "A2A AgentCard JSON (kind=agent) or ToolCard (kind=mcp_server)",
        }
    ),
]


def serialize_card(value: AgentCard | ToolCard | None) -> JsonObject | None:
    """Dump a card to its JSON object wire form."""
    if value is None:
        return None
    if isinstance(value, ToolCard):
        dumped: JsonObject = value.model_dump(mode="json")
        return dumped
    return agent_card_to_json_dict(value)


class _CardCarrier(BaseModel):
    """Shared card parsing/serialization for entry models carrying `kind`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _parse_card(cls, data: object) -> object:
        if isinstance(data, Mapping):
            card = data.get("card")
            if data.get("kind") == "agent" and isinstance(card, Mapping):
                data = {**data, "card": agent_card_from_dict(dict(card))}
        return data

    @field_serializer("card", check_fields=False)
    def _serialize_card(self, value: AgentCard | ToolCard | None) -> JsonObject | None:
        return serialize_card(value)


class RegistryEntry(_CardCarrier):
    """Card/metadata + tags + owner + health for one agent or MCP server.

    ``url`` is the gateway URL — the registry never stores internal URLs.
    """

    id: UUID
    kind: EntryKind
    card: Card
    url: str
    tags: list[str] = Field(default_factory=list)
    owner: str
    status: HealthStatus = "starting"
    last_seen: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def card_name(self) -> str:
        return str(self.card.name)


class RegistryEntryCreate(_CardCarrier):
    kind: EntryKind
    card: Card
    url: str
    tags: list[str] = Field(default_factory=list)


class RegistryEntryPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, arbitrary_types_allowed=True)

    card: Card | None = None
    url: str | None = None
    tags: list[str] | None = None

    @field_serializer("card")
    def _serialize_card(self, value: AgentCard | ToolCard | None) -> JsonObject | None:
        return serialize_card(value)

    @model_validator(mode="before")
    @classmethod
    def _parse_patch_card(cls, data: object) -> object:
        # A patch has no `kind`; a mapping card that does not look like a
        # ToolCard is parsed as an AgentCard (validated by the proto parser).
        if isinstance(data, Mapping):
            card = data.get("card")
            if isinstance(card, Mapping) and "tools" not in card:
                data = {**data, "card": agent_card_from_dict(dict(card))}
        return data


class Page(BaseModel):
    """One page of results."""

    model_config = ConfigDict(extra="forbid")

    items: list[RegistryEntry]
    total: int
    limit: int
    offset: int


class SearchQuery(BaseModel):
    """Registry search parameters (SPEC §3.3 / §5.1)."""

    model_config = ConfigDict(extra="forbid")

    q: str = ""
    tags: list[str] = Field(default_factory=list)
    kind: EntryKind | None = None
    status: HealthStatus | None = None
    semantic: bool = False
    owner: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class Capabilities(BaseModel):
    """Registry feature discovery (``GET /capabilities``)."""

    model_config = ConfigDict(extra="forbid")

    semantic_search: bool
    auth: AuthMode
    version: str


Severity = Literal["error", "warning"]


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Severity
    path: str
    message: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @classmethod
    def from_issues(cls, issues: list[ValidationIssue]) -> ValidationResult:
        return cls(valid=not any(i.severity == "error" for i in issues), issues=issues)


DefinitionStatus = Literal["draft", "deployed", "undeployed"]


class DefinitionInfo(BaseModel):
    """Summary of a named definition owned by the runtime (SPEC §6.1)."""

    model_config = ConfigDict(extra="forbid")

    name: Slug
    display_name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    expose_kind: Literal["a2a", "mcp"]
    status: DefinitionStatus
    latest_version: int | None = None
    deployed_version: int | None = None
    deployed_version_label: VersionLabel | None = None
    endpoint_url: str | None = None
    owner: str = ""
    created_at: datetime
    updated_at: datetime
    definition: FlowDefinition | None = None


class DeploymentInfo(BaseModel):
    """Result of a deploy (SPEC §6.1).

    ``version`` is the runtime's deploy counter and the version's identity
    (rollback, endpoint state). ``version_label`` is the publisher's optional
    semantic version for the same snapshot — it labels, it does not identify.
    """

    model_config = ConfigDict(extra="forbid")

    name: Slug
    version: int
    version_label: VersionLabel | None = None
    endpoint_url: str
    registry_id: UUID | None = None


__all__ = [
    "AuthMode",
    "Capabilities",
    "Card",
    "DefinitionInfo",
    "DefinitionStatus",
    "DeploymentInfo",
    "EntryKind",
    "HealthStatus",
    "Page",
    "RegistryEntry",
    "RegistryEntryCreate",
    "RegistryEntryPatch",
    "SearchQuery",
    "Severity",
    "ToolCard",
    "ToolCardTool",
    "ValidationIssue",
    "ValidationResult",
]
