"""v1.1 node types: router (conditional branching) and template (static text)."""

from __future__ import annotations

from typing import Any

from agentplane_core import (
    FlowDefinition,
    RouterNode,
    TemplateNode,
    input_ports,
    output_ports,
    validate_structure,
)


def _fallback_flow(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "name": "rag-with-fallback",
        "expose": {"kind": "a2a"},
        "nodes": [
            {
                "id": "start_1",
                "type": "start",
                "version": 1,
                "config": {
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    }
                },
            },
            {
                "id": "retrieve_1",
                "type": "retrieval",
                "version": 1,
                "config": {"resource": "kb", "collection": "docs"},
            },
            {
                "id": "check_1",
                "type": "router",
                "version": 1,
                "config": {
                    "input_type": "documents",
                    "rules": [{"when": "not_empty", "branch": "found"}],
                    "default_branch": "missing",
                },
            },
            {
                "id": "call_1",
                "type": "llm_call",
                "version": 1,
                "config": {"resource": "llm", "prompt": "Answer {query} using {documents}"},
            },
            {
                "id": "fallback_1",
                "type": "template",
                "version": 1,
                "config": {"text": "No information available."},
            },
            {"id": "end_1", "type": "end", "version": 1, "config": {"output_from": ""}},
        ],
        "edges": [
            {"from": "start_1.query", "to": "retrieve_1.query"},
            {"from": "start_1.query", "to": "call_1.query"},
            {"from": "retrieve_1.documents", "to": "check_1.input"},
            {"from": "check_1.found", "to": "call_1.documents"},
            {"from": "check_1.missing", "to": "fallback_1.trigger"},
            {"from": "call_1.text", "to": "end_1.input"},
            {"from": "fallback_1.text", "to": "end_1.input"},
        ],
    }
    base.update(overrides)
    return base


def _codes(defn: dict[str, Any]) -> set[str]:
    return {issue.code for issue in validate_structure(defn)}


def test_fallback_flow_validates_clean() -> None:
    issues = validate_structure(_fallback_flow())
    assert [i for i in issues if i.severity == "error"] == []


def test_router_ports_pass_the_input_type_through() -> None:
    defn = FlowDefinition.model_validate(_fallback_flow())
    router = next(n for n in defn.nodes if isinstance(n, RouterNode))
    assert input_ports(router) == {"input": "documents"}
    assert output_ports(router) == {"found": "documents", "missing": "documents"}


def test_template_has_trigger_and_var_ports() -> None:
    defn = FlowDefinition.model_validate(_fallback_flow())
    template = next(n for n in defn.nodes if isinstance(n, TemplateNode))
    assert input_ports(template) == {"trigger": "text"}
    assert output_ports(template) == {"text": "text"}

    flow = _fallback_flow()
    flow["nodes"][4]["config"]["text"] = "Nothing on {topic}."
    defn = FlowDefinition.model_validate(flow)
    template = next(n for n in defn.nodes if isinstance(n, TemplateNode))
    assert input_ports(template) == {"trigger": "text", "topic": "text"}


def test_empty_template_text_is_e010() -> None:
    flow = _fallback_flow()
    flow["nodes"][4]["config"]["text"] = ""
    assert "E010" in _codes(flow)


def test_duplicate_router_branches_are_e011() -> None:
    flow = _fallback_flow()
    flow["nodes"][2]["config"]["default_branch"] = "found"
    assert "E011" in _codes(flow)


def test_empty_output_from_requires_inbound_edge() -> None:
    # with the input wire: fine (branched flows feed end from either side)
    assert "E030" not in _codes(_fallback_flow())
    # without any inbound edge and no output_from: E030
    flow = _fallback_flow()
    flow["edges"] = [e for e in flow["edges"] if not e["to"].startswith("end_1")]
    assert "E030" in _codes(flow)


def test_roundtrip_with_v11_nodes() -> None:
    defn = FlowDefinition.model_validate(_fallback_flow())
    assert FlowDefinition.model_validate(defn.canonical_dict()) == defn
