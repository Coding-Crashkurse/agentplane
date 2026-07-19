"""Golden tests for stateless validation — these pin the E0xx contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from agentplane_core import ValidationResult, validate_structure

EXAMPLES_DIR = Path(__file__).parents[3] / "examples"


def _echo() -> dict[str, Any]:
    with (EXAMPLES_DIR / "echo-agent.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def _codes(defn: dict[str, Any]) -> list[tuple[str, str]]:
    return [(issue.code, issue.path) for issue in validate_structure(defn)]


def test_valid_examples_have_no_errors() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.yaml")):
        with path.open(encoding="utf-8") as fh:
            issues = validate_structure(yaml.safe_load(fh))
        errors = [i for i in issues if i.severity == "error"]
        assert not errors, f"{path.name}: {errors}"


def test_e001_unsupported_schema_version() -> None:
    defn = _echo()
    defn["schema_version"] = 99
    assert ("E001", "schema_version") in _codes(defn)


def test_e002_unknown_node_type() -> None:
    defn = _echo()
    defn["nodes"][1]["type"] = "quantum_leap"
    assert ("E002", "nodes/call_1/type") in _codes(defn)


def test_e002_unknown_node_version() -> None:
    defn = _echo()
    defn["nodes"][1]["version"] = 42
    assert ("E002", "nodes/call_1/version") in _codes(defn)


def test_e010_missing_required_field() -> None:
    defn = _echo()
    del defn["nodes"][1]["config"]["prompt"]
    assert ("E010", "nodes/call_1/config/prompt") in _codes(defn)


def test_e010_mcp_tool_without_target() -> None:
    defn = _echo()
    defn["nodes"].append(
        {"id": "tool_1", "type": "mcp_tool", "version": 1, "config": {"tool": "x"}}
    )
    defn["edges"].append({"from": "start_1.message", "to": "tool_1.q"})
    codes = [c for c, _ in _codes(defn)]
    assert "E010" in codes


def test_e011_invalid_field_type() -> None:
    defn = _echo()
    defn["nodes"][1]["config"]["stream"] = "definitely"
    assert ("E011", "nodes/call_1/config/stream") in _codes(defn)


def test_e023_credential_like_literal() -> None:
    defn = _echo()
    defn["nodes"][1]["config"]["system_prompt"] = "api_key=sk-abcdef0123456789abcdef"
    assert ("E023", "nodes/call_1/config/system_prompt") in _codes(defn)


def test_e030_end_unreachable() -> None:
    defn = _echo()
    defn["edges"] = [{"from": "call_1.text", "to": "end_1.input"}]
    assert ("E030", "nodes/end_1") in _codes(defn)


def test_e031_cycle() -> None:
    defn = _echo()
    defn["nodes"].append(
        {
            "id": "call_2",
            "type": "llm_call",
            "version": 1,
            "config": {"resource": "default-llm", "prompt": "{a}"},
        }
    )
    defn["edges"] += [
        {"from": "call_1.text", "to": "call_2.a"},
        {"from": "call_2.text", "to": "call_1.message"},
    ]
    assert "E031" in [c for c, _ in _codes(defn)]


def test_e032_dangling_edge() -> None:
    defn = _echo()
    defn["edges"].append({"from": "start_1.missing_port", "to": "call_1.message"})
    assert "E032" in [c for c, _ in _codes(defn)]


def test_e032_incompatible_port_types() -> None:
    defn = _echo()
    defn["nodes"].append(
        {
            "id": "retrieve_1",
            "type": "retrieval",
            "version": 1,
            "config": {"resource": "kb", "collection": "docs"},
        }
    )
    # documents -> json arg port is not allowed (documents render only to text)
    defn["nodes"].append(
        {
            "id": "tool_1",
            "type": "mcp_tool",
            "version": 1,
            "config": {"url": "http://gw/mcp/x", "tool": "t", "args": {"payload": "payload"}},
        }
    )
    defn["edges"] += [
        {"from": "start_1.message", "to": "retrieve_1.query"},
        {"from": "retrieve_1.documents", "to": "tool_1.payload"},
        {"from": "tool_1.result", "to": "call_1.message"},
    ]
    assert "E032" in [c for c, _ in _codes(defn)]


def test_e040_duplicates() -> None:
    defn = _echo()
    defn["nodes"].append(copy.deepcopy(defn["nodes"][1]))
    defn["edges"].append(copy.deepcopy(defn["edges"][0]))
    codes = _codes(defn)
    assert ("E040", "nodes/call_1") in codes
    assert any(code == "E040" and path.startswith("edges/") for code, path in codes)


def test_e050_mcp_without_tool_name() -> None:
    defn = _echo()
    defn["expose"] = {"kind": "mcp"}
    assert ("E050", "expose/tool_name") in _codes(defn)


def _text_router(rule: dict[str, Any], input_type: str = "text") -> dict[str, Any]:
    """Echo flow with a router between start and end carrying one rule."""
    defn = _echo()
    defn["nodes"].append(
        {
            "id": "route_1",
            "type": "router",
            "version": 1,
            "config": {"input_type": input_type, "rules": [rule], "default_branch": "otherwise"},
        }
    )
    defn["edges"] = [
        {"from": "start_1.message", "to": "route_1.input"},
        {"from": "route_1.hit", "to": "call_1.message"},
        {"from": "route_1.otherwise", "to": "call_1.message"},
        {"from": "call_1.text", "to": "end_1.input"},
    ]
    return defn


def test_e010_router_comparison_requires_value() -> None:
    defn = _text_router({"when": "equals", "branch": "hit"})
    assert ("E010", "nodes/route_1/config/rules/0/value") in _codes(defn)


def test_e011_router_numeric_condition_needs_number() -> None:
    defn = _text_router({"when": "gt", "value": "high", "branch": "hit"}, input_type="json")
    assert ("E011", "nodes/route_1/config/rules/0/value") in _codes(defn)


def test_e011_router_path_requires_json_input() -> None:
    defn = _text_router({"when": "equals", "path": "category", "value": "x", "branch": "hit"})
    assert ("E011", "nodes/route_1/config/rules/0/path") in _codes(defn)


def test_e011_router_documents_reject_comparisons() -> None:
    defn = _text_router({"when": "equals", "value": "x", "branch": "hit"}, input_type="documents")
    assert ("E011", "nodes/route_1/config/rules/0/when") in _codes(defn)


def test_e011_router_emptiness_takes_no_value() -> None:
    defn = _text_router({"when": "not_empty", "value": "x", "branch": "hit"})
    assert ("E011", "nodes/route_1/config/rules/0/value") in _codes(defn)


def test_router_json_comparison_rules_are_valid() -> None:
    defn = _text_router(
        {"when": "equals", "path": "category", "value": "billing", "branch": "hit"},
        input_type="json",
    )
    errors = [i for i in validate_structure(defn) if i.severity == "error"]
    assert not errors, errors


def _orchestrator(agents: list[dict[str, Any]]) -> dict[str, Any]:
    defn = _echo()
    defn["nodes"].append(
        {
            "id": "agent_1",
            "type": "agent",
            "version": 1,
            "config": {"resource": "default-llm", "prompt": "{q}", "agents": agents},
        }
    )
    defn["edges"] = [
        {"from": "start_1.message", "to": "agent_1.q"},
        {"from": "agent_1.text", "to": "end_1.input"},
    ]
    return defn


def test_e060_agent_self_reference() -> None:
    defn = _orchestrator([{"name": "echo-agent"}])
    assert ("E060", "nodes/agent_1/config/agents/0/name") in _codes(defn)


def test_e011_duplicate_agent_reference() -> None:
    defn = _orchestrator([{"name": "helper"}, {"name": "helper"}])
    assert ("E011", "nodes/agent_1/config/agents/1/name") in _codes(defn)


def test_agent_references_are_valid_structurally() -> None:
    defn = _orchestrator([{"name": "helper"}, {"name": "researcher"}])
    errors = [i for i in validate_structure(defn) if i.severity == "error"]
    assert not errors, errors


def test_w002_unused_node() -> None:
    defn = _echo()
    defn["nodes"].append(
        {
            "id": "zombie",
            "type": "retrieval",
            "version": 1,
            "config": {"resource": "kb", "collection": "docs"},
        }
    )
    defn["edges"].append({"from": "start_1.message", "to": "zombie.query"})
    issues = validate_structure(defn)
    assert ("W002", "nodes/zombie") in [(i.code, i.path) for i in issues]
    assert all(i.severity == "warning" for i in issues if i.code == "W002")


def test_w003_stream_ignored_for_mcp() -> None:
    defn = _echo()
    defn["expose"] = {"kind": "mcp", "tool_name": "echo"}
    result = ValidationResult.from_issues(validate_structure(defn))
    assert result.valid  # warnings only
    assert "W003" in [i.code for i in result.issues]


def test_validation_result_valid_flag() -> None:
    defn = _echo()
    assert ValidationResult.from_issues(validate_structure(defn)).valid
    defn["schema_version"] = 2
    assert not ValidationResult.from_issues(validate_structure(defn)).valid


def test_issues_are_deterministically_sorted() -> None:
    defn = _echo()
    defn["expose"] = {"kind": "mcp"}
    defn["edges"].append({"from": "start_1.missing", "to": "call_1.message"})
    first = validate_structure(copy.deepcopy(defn))
    second = validate_structure(copy.deepcopy(defn))
    assert [i.model_dump() for i in first] == [i.model_dump() for i in second]
    severities = [i.severity for i in first]
    assert severities == sorted(severities, key=lambda s: s == "warning")


def test_golden_files(tmp_path: Path) -> None:
    """Invalid golden definitions ship with their expected codes."""
    golden_dir = Path(__file__).parent / "golden"
    files = sorted(golden_dir.glob("*.yaml"))
    assert files, "golden directory must not be empty"
    for path in files:
        with path.open(encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
        expected = set(payload["expected_codes"])
        actual = {i.code for i in validate_structure(payload["definition"])}
        assert expected <= actual, f"{path.name}: expected {expected}, got {actual}"
