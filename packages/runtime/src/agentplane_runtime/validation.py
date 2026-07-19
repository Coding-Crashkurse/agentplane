"""Stateful validation (SPEC §3.7): core checks + E020/E021/E022/E061/E062/W001/W004.

The runtime's answer (``POST /definitions/validate``) is authoritative; it
runs ``agentplane_core.validation.validate_structure`` first — literally the
same code the builder uses locally — and adds the checks that need runtime
state (resources, dimensions, deprecations, registry agent references).
"""

from __future__ import annotations

from collections.abc import Mapping

from agentplane_core import (
    DEPRECATED_NODE_VERSIONS,
    AgentNode,
    FlowDefinition,
    LlmCallNode,
    McpToolNode,
    RerankNode,
    RetrievalNode,
    ValidationIssue,
    ValidationResult,
    VectorDBResource,
    validate_structure,
)
from agentplane_core.resources import VECTOR_DB_KINDS
from agentplane_runtime.directory import (
    AgentDirectory,
    AgentNotFoundError,
    AgentResolutionError,
)
from agentplane_runtime.resources import ResourceNotFoundError, ResourceService

_EXPECTED_KINDS = {
    "llm_call": frozenset({"model_provider"}),
    "retrieval": VECTOR_DB_KINDS,
    "mcp_tool": frozenset({"mcp_server"}),
    "rerank": frozenset({"model_provider"}),
    "agent": frozenset({"model_provider"}),
}


def _issue(code: str, severity: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue.model_validate(
        {"code": code, "severity": severity, "path": path, "message": message}
    )


async def _agent_tool_issues(node: AgentNode, resources: ResourceService) -> list[ValidationIssue]:
    """E020/E021 for each of an agent's tool resources (each must be an mcp_server)."""
    issues: list[ValidationIssue] = []
    for index, tool in enumerate(node.config.tools):
        path = f"nodes/{node.id}/config/tools/{index}/resource"
        try:
            resource = await resources.get_raw(tool.resource)
        except ResourceNotFoundError:
            issues.append(_issue("E020", "error", path, f"unknown resource {tool.resource!r}"))
            continue
        if resource.kind != "mcp_server":
            issues.append(
                _issue(
                    "E021",
                    "error",
                    path,
                    f"resource {tool.resource!r} has kind {resource.kind!r}; expected 'mcp_server'",
                )
            )
    return issues


async def agent_reference_issues(
    defn: FlowDefinition, agents: AgentDirectory | None
) -> list[ValidationIssue]:
    """E061/E062 for every sub-agent reference of the flow's agent nodes.

    Also called on deploy (fail-fast): a reference that validated against the
    registry at draft time may have been undeployed or disabled since.
    """
    issues: list[ValidationIssue] = []
    for node in defn.nodes:
        if not isinstance(node, AgentNode) or not node.config.agents:
            continue
        for index, ref in enumerate(node.config.agents):
            path = f"nodes/{node.id}/config/agents/{index}/name"
            if agents is None or not agents.enabled:
                issues.append(
                    _issue("E061", "error", path, "agent references require a configured registry")
                )
                continue
            try:
                resolved = await agents.resolve(ref.name)
            except AgentNotFoundError:
                issues.append(
                    _issue(
                        "E062",
                        "error",
                        path,
                        f"no enabled A2A agent {ref.name!r} in the registry",
                    )
                )
            except AgentResolutionError as exc:
                issues.append(_issue("E061", "error", path, str(exc)))
            else:
                if resolved.status == "unhealthy":
                    issues.append(
                        _issue(
                            "W004",
                            "warning",
                            path,
                            f"agent {ref.name!r} is currently unhealthy",
                        )
                    )
    return issues


async def validate_full(
    defn: FlowDefinition | Mapping[str, object],
    resources: ResourceService,
    agents: AgentDirectory | None = None,
) -> ValidationResult:
    """Structural checks (core) + resource/dimension/deprecation/agent-ref checks."""
    issues = list(validate_structure(defn))
    if any(i.severity == "error" for i in issues):
        return ValidationResult.from_issues(issues)

    parsed = defn if isinstance(defn, FlowDefinition) else FlowDefinition.model_validate(dict(defn))

    for node in parsed.nodes:
        if (node.type, node.version) in DEPRECATED_NODE_VERSIONS:
            issues.append(
                _issue(
                    "W001",
                    "warning",
                    f"nodes/{node.id}/version",
                    f"node version {node.version} of {node.type!r} is deprecated",
                )
            )
        if not isinstance(node, LlmCallNode | RetrievalNode | McpToolNode | RerankNode | AgentNode):
            continue
        resource_name = node.config.resource
        if resource_name is None:  # mcp_tool with direct url
            continue
        path = f"nodes/{node.id}/config/resource"
        try:
            resource = await resources.get_raw(resource_name)
        except ResourceNotFoundError:
            issues.append(_issue("E020", "error", path, f"unknown resource {resource_name!r}"))
            continue
        expected = _EXPECTED_KINDS[node.type]
        if resource.kind not in expected:
            issues.append(
                _issue(
                    "E021",
                    "error",
                    path,
                    f"resource {resource_name!r} has kind {resource.kind!r}; "
                    f"expected one of {sorted(expected)}",
                )
            )
            continue
        if isinstance(node, RetrievalNode) and isinstance(resource, VectorDBResource):
            e022 = await resources.check_collection_dimension(
                resource, node.config.collection, f"nodes/{node.id}/config/collection"
            )
            if e022 is not None:
                issues.append(e022)

        if isinstance(node, AgentNode):
            issues.extend(await _agent_tool_issues(node, resources))

    issues.extend(await agent_reference_issues(parsed, agents))
    return ValidationResult.from_issues(issues)


__all__ = ["agent_reference_issues", "validate_full"]
