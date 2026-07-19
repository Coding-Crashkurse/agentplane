"""The Definition schema — the platform's single artifact type (SPEC §3.1)."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentplane_core.types import (
    BranchName,
    JsonObject,
    JsonSchema,
    NodeId,
    PortType,
    Slug,
    ToolName,
    split_port_ref,
)

SCHEMA_VERSION = 1

# (type, version) pairs the platform knows how to execute. Unknown pairs fail
# validation with E002. Append-only per node-type version.
NODE_CATALOG: frozenset[tuple[str, int]] = frozenset(
    {
        ("start", 1),
        ("end", 1),
        ("llm_call", 1),
        ("mcp_tool", 1),
        ("retrieval", 1),
        # v1.1 additions
        ("router", 1),
        ("template", 1),
        ("rerank", 1),
        ("agent", 1),
    }
)

# (type, version) pairs that still run but warn with W001 on validation.
DEPRECATED_NODE_VERSIONS: frozenset[tuple[str, int]] = frozenset()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExposeConfig(_StrictModel):
    """How the runtime serves a flow: as an A2A agent or as an MCP server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["a2a", "mcp"]
    tool_name: ToolName | None = None
    tool_description: str = ""
    examples: list[str] = Field(
        default_factory=list,
        description=(
            "Example prompts for this flow. Served as `skill.examples` on the A2A "
            "agent card so calling agents can route to it."
        ),
    )


class StartNodeConfig(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_schema: JsonSchema


class EndNodeConfig(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_from: str = Field(
        default="",
        description=(
            "Reference 'node_id.port' the flow output is read from. Empty means: "
            "take the value wired into the `input` port — required for branched "
            "flows where different runs feed `end` from different nodes."
        ),
    )


class LlmCallNodeConfig(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: Slug
    model: str = ""
    prompt: str
    system_prompt: str = ""
    structured_output: JsonSchema | None = None
    stream: bool = False
    history: bool = Field(
        default=False,
        description=(
            "Prepend the conversation history as chat messages. The history is "
            "the prior turns of the caller's conversation (A2A contextId), "
            "loaded by the runtime from its task store; with history disabled "
            "or no persistence the node sees only the current prompt."
        ),
    )
    history_max_turns: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Cap on prior exchanges (user + assistant pairs) fed to the model.",
    )


class AgentToolRef(_StrictModel):
    """One tool an agent may call: an MCP server resource, optionally narrowed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: Slug = Field(description="McpServer resource the tool(s) live on")
    tool: str = Field(
        default="", description="Specific tool name; empty = every tool the server exposes"
    )


class AgentNodeConfig(_StrictModel):
    """LLM + tool loop (v1.1): the model calls MCP tools until it answers.

    The prompt's ``{vars}`` become input ports (like ``llm_call``). Each turn the
    model may emit tool calls (OpenAI tool-calling); the runtime runs them via
    MCP and feeds the results back, up to ``max_iterations``, then returns the
    final assistant text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: Slug = Field(description="ModelProvider resource")
    model: str = ""
    prompt: str
    system_prompt: str = ""
    tools: list[AgentToolRef] = Field(default_factory=list)
    max_iterations: int = Field(
        default=6, ge=1, le=50, description="Cap on tool-call turns before forcing a final answer."
    )


class McpToolNodeConfig(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: Slug | None = None
    url: str | None = None
    tool: str
    args: dict[str, str] = Field(
        default_factory=dict, description="input port name -> tool argument name"
    )


class RetrievalNodeConfig(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: Slug
    collection: str
    top_k: int = Field(default=4, ge=1, le=100)
    filter: JsonObject | None = None
    min_score: float | None = Field(
        default=None,
        description=(
            "Similarity score threshold. Hits below it are dropped, so a filled "
            "collection can still yield zero documents — which is what makes an "
            "empty-result router branch fire. Score semantics follow the "
            "collection's distance metric (cosine: 1.0 = identical)."
        ),
    )


class RerankNodeConfig(_StrictModel):
    """Reorder retrieved documents by relevance to the query (v1.1).

    The retrieval node returns a similarity top-k; a reranker (cross-encoder)
    then scores each document against the query directly and keeps the best
    ``top_n`` — the standard quality step between retrieval and the LLM. Reuses
    a ``model_provider`` resource whose ``base_url`` serves a ``/rerank``
    endpoint (Cohere/Jina/TEI-style); no new resource kind.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: Slug
    model: str = ""
    top_n: int = Field(default=4, ge=1, le=100)
    min_score: float | None = Field(
        default=None,
        description=(
            "Rerank score threshold; documents scoring below it are dropped, so a "
            "reranked set can be empty (which lets an empty-result router branch fire). "
            "Score scale is the reranker's own relevance score, not the retrieval metric."
        ),
    )


RouterInputType = Literal["text", "json", "documents"]


class RouterRule(_StrictModel):
    """One branch condition, evaluated in order; first match wins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    when: Literal["not_empty", "empty"]
    branch: BranchName


class RouterNodeConfig(_StrictModel):
    """Conditional branching (v1.1): routes the input value to one branch.

    Every rule branch plus the default branch becomes a typed output port
    that passes the input value through; only the chosen branch's downstream
    nodes execute.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_type: RouterInputType = "text"
    rules: list[RouterRule] = Field(min_length=1)
    default_branch: BranchName = "otherwise"


class TemplateNodeConfig(_StrictModel):
    """Static/interpolated text (v1.1): `{vars}` become input ports.

    The `trigger` input port exists on every template so a router branch can
    activate it even when the text needs no variables (fallback messages).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str


class _NodeBase(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NodeId
    version: int = Field(ge=1)


class StartNode(_NodeBase):
    type: Literal["start"]
    config: StartNodeConfig


class EndNode(_NodeBase):
    type: Literal["end"]
    config: EndNodeConfig


class LlmCallNode(_NodeBase):
    type: Literal["llm_call"]
    config: LlmCallNodeConfig


class AgentNode(_NodeBase):
    type: Literal["agent"]
    config: AgentNodeConfig


class McpToolNode(_NodeBase):
    type: Literal["mcp_tool"]
    config: McpToolNodeConfig


class RetrievalNode(_NodeBase):
    type: Literal["retrieval"]
    config: RetrievalNodeConfig


class RerankNode(_NodeBase):
    type: Literal["rerank"]
    config: RerankNodeConfig


class RouterNode(_NodeBase):
    type: Literal["router"]
    config: RouterNodeConfig


class TemplateNode(_NodeBase):
    type: Literal["template"]
    config: TemplateNodeConfig


Node = Annotated[
    StartNode
    | EndNode
    | LlmCallNode
    | AgentNode
    | McpToolNode
    | RetrievalNode
    | RerankNode
    | RouterNode
    | TemplateNode,
    Field(discriminator="type"),
]


class Edge(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    from_: str = Field(alias="from", serialization_alias="from")
    to: str


class LayoutPosition(BaseModel):
    """Canvas position — builder-owned, ignored by the runtime."""

    model_config = ConfigDict(extra="allow", frozen=True)

    x: float = 0
    y: float = 0


class Layout(BaseModel):
    """Canvas layout block — builder-owned, ignored by the runtime."""

    model_config = ConfigDict(extra="allow", frozen=True)

    nodes: dict[str, LayoutPosition] = Field(default_factory=dict)


class FlowDefinition(_StrictModel):
    """Versioned, serializable description of a flow (SPEC §3.1).

    Serialization is deterministic: nodes are sorted by id, edges by
    (from, to), keys stay in model order — ``parse(serialize(d)) == d``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int
    name: Slug
    display_name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    expose: ExposeConfig
    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)
    layout: Layout | None = None

    @model_validator(mode="before")
    @classmethod
    def _canonical_order(cls, data: object) -> object:
        """Sort nodes by id and edges by (from, to) for deterministic serialization."""
        if not isinstance(data, dict):
            return data

        def _node_key(item: object) -> str:
            if isinstance(item, dict):
                node_id = item.get("id", "")
                return node_id if isinstance(node_id, str) else ""
            return getattr(item, "id", "")

        def _edge_key(item: object) -> tuple[str, str]:
            if isinstance(item, dict):
                from_ = item.get("from", item.get("from_", ""))
                to = item.get("to", "")
                return (from_ if isinstance(from_, str) else "", to if isinstance(to, str) else "")
            return (getattr(item, "from_", ""), getattr(item, "to", ""))

        nodes = data.get("nodes")
        if isinstance(nodes, list):
            data = {**data, "nodes": sorted(nodes, key=_node_key)}
        edges = data.get("edges")
        if isinstance(edges, list):
            data = {**data, "edges": sorted(edges, key=_edge_key)}
        return data

    def node_map(self) -> dict[str, Node]:
        """Nodes keyed by id (last one wins on duplicates; E040 catches those)."""
        return {node.id: node for node in self.nodes}

    def canonical_dict(self) -> JsonObject:
        """Deterministic JSON-mode dump used for export and hashing."""
        dumped: JsonObject = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        return dumped


# `{vars}` in prompts become named input ports of that node. Double braces
# (`{{literal}}`) are not ports.
_PROMPT_VAR_RE = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}(?!\})")


def prompt_variables(template: str) -> list[str]:
    """Extract `{var}` input ports from a prompt template, in first-seen order."""
    seen: dict[str, None] = {}
    for match in _PROMPT_VAR_RE.finditer(template):
        seen.setdefault(match.group(1))
    return list(seen)


def _schema_property_port_type(prop_schema: object) -> PortType:
    if isinstance(prop_schema, dict) and prop_schema.get("type") == "string":
        return "text"
    return "json"


def output_ports(node: Node) -> dict[str, PortType]:  # noqa: PLR0911 - one return per node type
    """Typed output ports a node exposes."""
    match node:
        case StartNode():
            props = node.config.input_schema.get("properties")
            if not isinstance(props, dict):
                return {}
            return {name: _schema_property_port_type(schema) for name, schema in props.items()}
        case EndNode():
            return {}
        case LlmCallNode():
            ports: dict[str, PortType] = {"text": "text"}
            if node.config.structured_output is not None:
                ports["json"] = "json"
            return ports
        case AgentNode():
            return {"text": "text"}
        case McpToolNode():
            return {"result": "json"}
        case RetrievalNode():
            return {"documents": "documents"}
        case RerankNode():
            return {"documents": "documents"}
        case RouterNode():
            branches: dict[str, PortType] = {
                rule.branch: node.config.input_type for rule in node.config.rules
            }
            branches[node.config.default_branch] = node.config.input_type
            return branches
        case TemplateNode():
            return {"text": "text"}


def input_ports(node: Node) -> dict[str, PortType]:  # noqa: PLR0911 - one return per node type
    """Typed input ports a node accepts."""
    match node:
        case StartNode():
            return {}
        case EndNode():
            return {"input": "text"}
        case LlmCallNode():
            names = prompt_variables(node.config.prompt) + prompt_variables(
                node.config.system_prompt
            )
            return dict.fromkeys(names, "text")
        case AgentNode():
            names = prompt_variables(node.config.prompt) + prompt_variables(
                node.config.system_prompt
            )
            return dict.fromkeys(names, "text")
        case McpToolNode():
            return dict.fromkeys(node.config.args, "json")
        case RetrievalNode():
            return {"query": "text"}
        case RerankNode():
            return {"query": "text", "documents": "documents"}
        case RouterNode():
            return {"input": node.config.input_type}
        case TemplateNode():
            template_ports: dict[str, PortType] = {"trigger": "text"}
            for name in prompt_variables(node.config.text):
                template_ports[name] = "text"
            return template_ports


def ports_compatible(src: PortType, dst: PortType) -> bool:
    """Port type compatibility matrix (SPEC §3.1).

    Identical types always connect. ``documents -> text`` is rendered
    implicitly (concatenated chunks with source headers); ``json -> text``
    is stringified; ``text -> json`` is a plain JSON string value.
    """
    if src == dst:
        return True
    return (src, dst) in {("documents", "text"), ("json", "text"), ("text", "json")}


def single_required_string_input(defn: FlowDefinition) -> str | None:
    """Name of the single required string property of ``start.input_schema``, if any.

    Used for A2A input binding: plain message text binds to this property
    (SPEC §6.4); otherwise the message must be a JSON object.
    """
    start = next((n for n in defn.nodes if isinstance(n, StartNode)), None)
    if start is None:
        return None
    schema = start.config.input_schema
    required = schema.get("required")
    props = schema.get("properties")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(props, dict):
        return None
    name = required[0]
    if not isinstance(name, str):
        return None
    return name if _schema_property_port_type(props.get(name)) == "text" else None


__all__ = [
    "DEPRECATED_NODE_VERSIONS",
    "NODE_CATALOG",
    "SCHEMA_VERSION",
    "AgentNode",
    "AgentNodeConfig",
    "AgentToolRef",
    "Edge",
    "EndNode",
    "EndNodeConfig",
    "ExposeConfig",
    "FlowDefinition",
    "Layout",
    "LayoutPosition",
    "LlmCallNode",
    "LlmCallNodeConfig",
    "McpToolNode",
    "McpToolNodeConfig",
    "Node",
    "RerankNode",
    "RerankNodeConfig",
    "RetrievalNode",
    "RetrievalNodeConfig",
    "RouterInputType",
    "RouterNode",
    "RouterNodeConfig",
    "RouterRule",
    "StartNode",
    "StartNodeConfig",
    "TemplateNode",
    "TemplateNodeConfig",
    "input_ports",
    "output_ports",
    "ports_compatible",
    "prompt_variables",
    "single_required_string_input",
    "split_port_ref",
]
