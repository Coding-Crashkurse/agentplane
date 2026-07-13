"""Shared primitive types used across the agentplane contract."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# A JSON Schema document is open-ended by definition; ``Any`` is unavoidable here
# and every consumer treats it as an opaque, already-valid JSON object.
JsonSchema = dict[str, Any]

# A JSON object with arbitrary (but JSON-serializable) values.
JsonObject = dict[str, JsonValue]

# Port types a node can expose. Edges connect compatible port types only.
PortType = Literal["text", "json", "message", "documents"]

SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{1,62}$"
NODE_ID_PATTERN = r"^[a-z][a-z0-9_]*$"
TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
BRANCH_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
PORT_REF_PATTERN = r"^[a-z][a-z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$"

Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]
NodeId = Annotated[str, Field(pattern=NODE_ID_PATTERN)]
ToolName = Annotated[str, Field(pattern=TOOL_NAME_PATTERN)]
BranchName = Annotated[str, Field(pattern=BRANCH_NAME_PATTERN)]
PortRef = Annotated[str, Field(pattern=PORT_REF_PATTERN, description="Reference 'node_id.port'")]


def split_port_ref(ref: str) -> tuple[str, str]:
    """Split a ``node_id.port`` reference into its two parts."""
    node_id, _, port = ref.partition(".")
    return node_id, port


class Document(BaseModel):
    """One retrieved chunk flowing over a ``documents`` port."""

    model_config = ConfigDict(frozen=True)

    text: str
    score: float | None = None
    metadata: JsonObject = Field(default_factory=dict)


def render_documents(documents: list[Document]) -> str:
    """Implicit ``documents -> text`` rendering: concatenated chunks with source headers."""
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", f"document {i}")
        parts.append(f"[{source}]\n{doc.text}")
    return "\n\n".join(parts)
