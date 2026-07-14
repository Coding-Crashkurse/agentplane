"""Stateful validation (SPEC §3.7): core checks + E020/E021/E022/W001.

The runtime's answer (``POST /definitions/validate``) is authoritative; it
runs ``agentplane_core.validation.validate_structure`` first — literally the
same code the builder uses locally — and adds the checks that need runtime
state (resources, dimensions, deprecations).
"""

from __future__ import annotations

from collections.abc import Mapping

from agentplane_core import (
    DEPRECATED_NODE_VERSIONS,
    FlowDefinition,
    LlmCallNode,
    McpToolNode,
    RetrievalNode,
    ValidationIssue,
    ValidationResult,
    VectorDBResource,
    validate_structure,
)
from agentplane_core.resources import VECTOR_DB_KINDS
from agentplane_runtime.resources import ResourceNotFoundError, ResourceService

_EXPECTED_KINDS = {
    "llm_call": frozenset({"model_provider"}),
    "retrieval": VECTOR_DB_KINDS,
    "mcp_tool": frozenset({"mcp_server"}),
}


def _issue(code: str, severity: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue.model_validate(
        {"code": code, "severity": severity, "path": path, "message": message}
    )


async def validate_full(
    defn: FlowDefinition | Mapping[str, object], resources: ResourceService
) -> ValidationResult:
    """Structural checks (core) + resource/dimension/deprecation checks."""
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
        if not isinstance(node, LlmCallNode | RetrievalNode | McpToolNode):
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

    return ValidationResult.from_issues(issues)


__all__ = ["validate_full"]
