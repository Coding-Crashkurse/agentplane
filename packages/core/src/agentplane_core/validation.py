"""Stateless definition validation (SPEC §3.7).

``validate_structure`` implements every check that needs only the definition
itself: E001, E002, E010, E011, E023, E030, E031, E032, E040, E050, W002,
W003. Pure function, no I/O — importable by the runtime AND by external
consumers, so pre-deploy validation and deploy validation are the same code.

Checks that require runtime state (E020, E021, E022, W001) live in the
runtime and are exposed via ``POST /definitions/validate``. Local validation
is advisory; the runtime's answer is authoritative.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from graphlib import CycleError, TopologicalSorter
from typing import Literal

from pydantic import ValidationError

from agentplane_core.definition import (
    NODE_CATALOG,
    SCHEMA_VERSION,
    EndNode,
    FlowDefinition,
    LlmCallNode,
    McpToolNode,
    Node,
    RouterNode,
    StartNode,
    TemplateNode,
    input_ports,
    output_ports,
    ports_compatible,
)
from agentplane_core.registry import ValidationIssue
from agentplane_core.types import split_port_ref

# Stable error code registry (SPEC §3.5) — append-only, never change meanings.
ErrorCode = Literal[
    "E001",  # Unsupported schema_version
    "E002",  # Unknown node type/version
    "E010",  # Required field empty/missing
    "E011",  # Field type/format invalid
    "E020",  # Unknown resource reference (runtime)
    "E021",  # Resource kind mismatch for node (runtime)
    "E022",  # Embedding dimension mismatch (runtime)
    "E023",  # Credential-like literal in definition
    "E030",  # Terminal node unreachable from start / no inbound to end
    "E031",  # Cycle detected
    "E032",  # Dangling edge or incompatible port types
    "E040",  # Duplicate node id / duplicate edge
    "E050",  # Invalid expose config
    "W001",  # Node version deprecated (runtime)
    "W002",  # Unused node (no path to end)
    "W003",  # stream: true ignored for MCP-exposed flows
]

# Patterns that look like credentials; matching literals raise E023 because
# resources must be referenced by name only, never inlined.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ"),
    re.compile(r"(?i)\b(api[_-]?key|apikey|secret|password|access[_-]?token)\b\s*[:=]\s*\S{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{16,}"),
)


def _issue(code: ErrorCode, path: str, message: str) -> ValidationIssue:
    severity: Literal["error", "warning"] = "warning" if code.startswith("W") else "error"
    return ValidationIssue(code=code, severity=severity, path=path, message=message)


def _loc_to_path(loc: tuple[int | str, ...], raw_nodes: object) -> str:
    """Turn a pydantic error loc into a JSON-pointer-ish locator using node ids."""
    parts: list[str] = []
    i = 0
    while i < len(loc):
        part = loc[i]
        # Replace a node index with the node id when available.
        if parts and parts[-1] == "nodes" and isinstance(part, int):
            node_id: str | None = None
            if isinstance(raw_nodes, list) and part < len(raw_nodes):
                candidate = raw_nodes[part]
                if isinstance(candidate, Mapping) and isinstance(candidate.get("id"), str):
                    node_id = str(candidate["id"])
            parts.append(node_id if node_id is not None else str(part))
            # Skip the discriminator tag pydantic inserts for tagged unions.
            if (
                i + 1 < len(loc)
                and isinstance(loc[i + 1], str)
                and loc[i + 1]
                not in (
                    "id",
                    "type",
                    "version",
                    "config",
                )
            ):
                nxt = loc[i + 1]
                if isinstance(nxt, str) and nxt in {t for t, _ in NODE_CATALOG}:
                    i += 1
        else:
            parts.append(str(part))
        i += 1
    return "/".join(parts)


def _pydantic_issues(exc: ValidationError, raw: Mapping[str, object]) -> list[ValidationIssue]:
    # Error locs refer to the canonically sorted node list (the before-validator
    # sorts nodes by id), so resolve indices against the same ordering.
    raw_nodes = raw.get("nodes")
    if isinstance(raw_nodes, list):

        def _key(n: object) -> str:
            node_id = n.get("id") if isinstance(n, Mapping) else None
            return node_id if isinstance(node_id, str) else ""

        raw_nodes = sorted(raw_nodes, key=_key)
    issues: list[ValidationIssue] = []
    for err in exc.errors():
        loc = tuple(err["loc"])
        path = _loc_to_path(loc, raw_nodes)
        code: ErrorCode = "E010" if err["type"] == "missing" else "E011"
        issues.append(_issue(code, path, err["msg"]))
    return issues


def _precheck_raw(raw: Mapping[str, object]) -> list[ValidationIssue]:
    """Checks on the raw mapping that produce nicer codes than pydantic's."""
    issues: list[ValidationIssue] = []
    version = raw.get("schema_version")
    if version is None:
        issues.append(_issue("E010", "schema_version", "schema_version is required"))
    elif not isinstance(version, int) or isinstance(version, bool):
        issues.append(_issue("E011", "schema_version", "schema_version must be an integer"))
    elif version != SCHEMA_VERSION:
        issues.append(_issue("E001", "schema_version", f"unsupported schema_version {version}"))

    nodes = raw.get("nodes")
    known_types = {t for t, _ in NODE_CATALOG}
    if isinstance(nodes, list):
        for idx, node in enumerate(nodes):
            if not isinstance(node, Mapping):
                continue
            node_id = node.get("id")
            label = node_id if isinstance(node_id, str) else str(idx)
            ntype = node.get("type")
            nversion = node.get("version")
            if isinstance(ntype, str) and ntype not in known_types:
                issues.append(_issue("E002", f"nodes/{label}/type", f"unknown node type {ntype!r}"))
            elif (
                isinstance(ntype, str)
                and isinstance(nversion, int)
                and (ntype, nversion) not in NODE_CATALOG
            ):
                issues.append(
                    _issue(
                        "E002",
                        f"nodes/{label}/version",
                        f"unknown version {nversion} for node type {ntype!r}",
                    )
                )
    return issues


def _scan_secrets(defn: FlowDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def scan(value: object, path: str) -> None:
        if isinstance(value, str):
            for pattern in _SECRET_PATTERNS:
                if pattern.search(value):
                    issues.append(
                        _issue(
                            "E023",
                            path,
                            "credential-like literal found; reference resources by name",
                        )
                    )
                    return
        elif isinstance(value, Mapping):
            for key, item in value.items():
                scan(item, f"{path}/{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                scan(item, f"{path}/{i}")

    for node in defn.nodes:
        scan(node.config.model_dump(mode="json"), f"nodes/{node.id}/config")
    return issues


def _validate_expose(defn: FlowDefinition, start: StartNode | None) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if defn.expose.kind == "mcp":
        if not defn.expose.tool_name:
            issues.append(
                _issue("E050", "expose/tool_name", "kind 'mcp' requires expose.tool_name")
            )
        input_schema = start.config.input_schema if start is not None else None
        if start is not None and not isinstance(
            input_schema.get("properties") if input_schema else None, dict
        ):
            issues.append(
                _issue(
                    "E050",
                    f"nodes/{start.id}/config/input_schema",
                    "MCP-exposed flows need an object input_schema with properties",
                )
            )
    return issues


def _check_duplicates(defn: FlowDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    for node in defn.nodes:
        if node.id in seen_ids:
            issues.append(_issue("E040", f"nodes/{node.id}", f"duplicate node id {node.id!r}"))
        seen_ids.add(node.id)
    seen_edges: set[tuple[str, str]] = set()
    for idx, edge in enumerate(defn.edges):
        key = (edge.from_, edge.to)
        if key in seen_edges:
            issues.append(
                _issue("E040", f"edges/{idx}", f"duplicate edge {edge.from_} -> {edge.to}")
            )
        seen_edges.add(key)
    return issues


def _check_edges(
    defn: FlowDefinition, nodes: dict[str, Node], node_edges: dict[str, set[str]]
) -> list[ValidationIssue]:
    """E032 — dangling edges and port type compatibility. Fills ``node_edges``."""
    issues: list[ValidationIssue] = []
    for idx, edge in enumerate(defn.edges):
        src_node_id, src_port = split_port_ref(edge.from_)
        dst_node_id, dst_port = split_port_ref(edge.to)
        path = f"edges/{idx}"
        dangling = False
        for node_id, port, port_map_fn, role in (
            (src_node_id, src_port, output_ports, "source"),
            (dst_node_id, dst_port, input_ports, "target"),
        ):
            target = nodes.get(node_id)
            if target is None:
                issues.append(
                    _issue("E032", path, f"dangling edge: unknown {role} node {node_id!r}")
                )
                dangling = True
            elif port not in port_map_fn(target):
                issues.append(
                    _issue(
                        "E032",
                        path,
                        f"dangling edge: {role} node {node_id!r} has no port {port!r}",
                    )
                )
                dangling = True
        if dangling:
            continue
        src_type = output_ports(nodes[src_node_id])[src_port]
        dst_type = input_ports(nodes[dst_node_id])[dst_port]
        if not ports_compatible(src_type, dst_type):
            issues.append(
                _issue(
                    "E032",
                    path,
                    f"incompatible port types: {edge.from_} ({src_type}) -> {edge.to} ({dst_type})",
                )
            )
        node_edges[src_node_id].add(dst_node_id)
    for end in (n for n in defn.nodes if isinstance(n, EndNode)):
        # Empty output_from means "take the value wired into `input`" —
        # required for branched flows where runs feed `end` from different
        # nodes; only a non-empty reference is checked.
        if not end.config.output_from:
            continue
        src_node_id, src_port = split_port_ref(end.config.output_from)
        src = nodes.get(src_node_id)
        if src is None or src_port not in output_ports(src):
            issues.append(
                _issue(
                    "E032",
                    f"nodes/{end.id}/config/output_from",
                    f"output_from references unknown port {end.config.output_from!r}",
                )
            )
    return issues


def _check_reachability(
    nodes: dict[str, Node],
    node_edges: dict[str, set[str]],
    predecessors: dict[str, set[str]],
    start: StartNode,
    end: EndNode,
) -> list[ValidationIssue]:
    """E030 reachability + W002 unused nodes (assumes a DAG)."""
    issues: list[ValidationIssue] = []
    reachable: set[str] = set()
    stack = [start.id]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(node_edges.get(current, ()))
    inbound_to_end = predecessors[end.id] | (
        {split_port_ref(end.config.output_from)[0]} & set(nodes)
    )
    if end.id not in reachable and not (inbound_to_end & reachable):
        issues.append(_issue("E030", f"nodes/{end.id}", "end node is unreachable from start"))
    if not inbound_to_end:
        issues.append(_issue("E030", f"nodes/{end.id}", "end node has no inbound edge"))

    reaches_end: set[str] = {end.id} | inbound_to_end
    changed = True
    while changed:
        changed = False
        for src_id, dst_ids in node_edges.items():
            if src_id not in reaches_end and (dst_ids & reaches_end):
                reaches_end.add(src_id)
                changed = True
    for node_id in nodes:
        if node_id not in reaches_end:
            issues.append(_issue("W002", f"nodes/{node_id}", "unused node: no path to end"))
    return issues


def _validate_graph(defn: FlowDefinition) -> list[ValidationIssue]:
    nodes = defn.node_map()
    issues = _check_duplicates(defn)

    starts = [n for n in defn.nodes if isinstance(n, StartNode)]
    ends = [n for n in defn.nodes if isinstance(n, EndNode)]
    if len(starts) != 1:
        issues.append(
            _issue("E030", "nodes", f"exactly one start node required, found {len(starts)}")
        )
    if len(ends) != 1:
        issues.append(_issue("E030", "nodes", f"exactly one end node required, found {len(ends)}"))

    node_edges: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    issues.extend(_check_edges(defn, nodes, node_edges))

    # E031 — cycles (node-level graph)
    predecessors: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    for src_id, dst_ids in node_edges.items():
        for dst_id in dst_ids:
            predecessors[dst_id].add(src_id)
    try:
        TopologicalSorter(predecessors).prepare()
    except CycleError as exc:
        cycle = " -> ".join(str(n) for n in exc.args[1])
        issues.append(_issue("E031", "edges", f"cycle detected: {cycle}"))
        return issues  # reachability below assumes a DAG

    if len(starts) == 1 and len(ends) == 1:
        issues.extend(_check_reachability(nodes, node_edges, predecessors, starts[0], ends[0]))
    return issues


def _validate_node_fields(defn: FlowDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for node in defn.nodes:
        if (node.type, node.version) not in NODE_CATALOG:
            issues.append(
                _issue(
                    "E002",
                    f"nodes/{node.id}/version",
                    f"unknown version {node.version} for node type {node.type!r}",
                )
            )
        if isinstance(node, McpToolNode):
            if not node.config.resource and not node.config.url:
                issues.append(
                    _issue(
                        "E010",
                        f"nodes/{node.id}/config",
                        "mcp_tool needs either 'resource' or 'url'",
                    )
                )
            if not node.config.tool:
                issues.append(
                    _issue("E010", f"nodes/{node.id}/config/tool", "tool must not be empty")
                )
        if isinstance(node, LlmCallNode) and not node.config.prompt:
            issues.append(
                _issue("E010", f"nodes/{node.id}/config/prompt", "prompt must not be empty")
            )
        if isinstance(node, StartNode):
            props = node.config.input_schema.get("properties")
            if props is not None and not isinstance(props, dict):
                issues.append(
                    _issue(
                        "E011",
                        f"nodes/{node.id}/config/input_schema/properties",
                        "properties must be an object",
                    )
                )
        if isinstance(node, TemplateNode) and not node.config.text:
            issues.append(_issue("E010", f"nodes/{node.id}/config/text", "text must not be empty"))
        if isinstance(node, RouterNode):
            branches = [rule.branch for rule in node.config.rules]
            branches.append(node.config.default_branch)
            duplicates = {name for name in branches if branches.count(name) > 1}
            if duplicates:
                issues.append(
                    _issue(
                        "E011",
                        f"nodes/{node.id}/config/rules",
                        "branch names must be unique (incl. default_branch): "
                        + ", ".join(sorted(duplicates)),
                    )
                )
    return issues


def _validate_stream_warning(defn: FlowDefinition) -> list[ValidationIssue]:
    if defn.expose.kind != "mcp":
        return []
    return [
        _issue(
            "W003",
            f"nodes/{node.id}/config/stream",
            "stream: true is ignored for MCP-exposed flows",
        )
        for node in defn.nodes
        if isinstance(node, LlmCallNode) and node.config.stream
    ]


def validate_structure(
    defn: FlowDefinition | Mapping[str, object],
) -> list[ValidationIssue]:
    """Run all stateless checks; returns issues sorted by (path, code)."""
    if isinstance(defn, FlowDefinition):
        parsed = defn
        issues: list[ValidationIssue] = []
        if parsed.schema_version != SCHEMA_VERSION:
            issues.append(
                _issue(
                    "E001",
                    "schema_version",
                    f"unsupported schema_version {parsed.schema_version}",
                )
            )
    else:
        issues = _precheck_raw(defn)
        if any(i.severity == "error" for i in issues):
            return _sorted(issues)
        try:
            parsed = FlowDefinition.model_validate(dict(defn))
        except ValidationError as exc:
            issues.extend(_pydantic_issues(exc, defn))
            return _sorted(issues)

    start = next((n for n in parsed.nodes if isinstance(n, StartNode)), None)
    issues.extend(_validate_node_fields(parsed))
    issues.extend(_validate_expose(parsed, start))
    issues.extend(_validate_graph(parsed))
    issues.extend(_scan_secrets(parsed))
    issues.extend(_validate_stream_warning(parsed))
    return _sorted(issues)


def _sorted(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return sorted(issues, key=lambda i: (i.severity == "warning", i.path, i.code))


__all__ = ["ErrorCode", "validate_structure"]
