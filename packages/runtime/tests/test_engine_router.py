"""Router/template execution (v1.1): only the chosen branch runs."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agentplane_core import FlowDefinition, RouterNodeConfig
from agentplane_runtime.engine import ExecutionContext, FlowRunner, _route
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.settings import RuntimeSettings


def _fallback_flow() -> FlowDefinition:
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "route-demo",
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
                "id": "check_1",
                "type": "router",
                "version": 1,
                "config": {
                    "input_type": "text",
                    "rules": [{"when": "not_empty", "branch": "found"}],
                    "default_branch": "missing",
                },
            },
            {
                "id": "shout_1",
                "type": "template",
                "version": 1,
                "config": {"text": "You said: {trigger}"},
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
            {"from": "start_1.query", "to": "check_1.input"},
            {"from": "check_1.found", "to": "shout_1.trigger"},
            {"from": "check_1.missing", "to": "fallback_1.trigger"},
            {"from": "shout_1.text", "to": "end_1.input"},
            {"from": "fallback_1.text", "to": "end_1.input"},
        ],
    }
    return FlowDefinition.model_validate(data)


def _runner() -> FlowRunner:
    # Resources are never touched by start/router/template/end nodes.
    context = ExecutionContext(
        resources=cast(ResourceService, object()), settings=RuntimeSettings(_env_file=None)
    )
    return FlowRunner(_fallback_flow(), context)


async def test_router_takes_the_matching_branch() -> None:
    result = await _runner().execute({"query": "hello"})
    assert result == "You said: hello"


async def test_router_falls_back_to_default_branch() -> None:
    result = await _runner().execute({"query": "   "})
    assert result == "No information available."


def _config(rules: list[dict[str, Any]], input_type: str = "json") -> RouterNodeConfig:
    return RouterNodeConfig.model_validate(
        {"input_type": input_type, "rules": rules, "default_branch": "otherwise"}
    )


@pytest.mark.parametrize(
    ("rule", "value", "expected"),
    [
        (
            {"when": "equals", "path": "category", "value": "billing", "branch": "hit"},
            {"category": "billing"},
            "hit",
        ),
        (
            {"when": "equals", "path": "category", "value": "billing", "branch": "hit"},
            {"category": "technical"},
            "otherwise",
        ),
        (
            {"when": "not_equals", "path": "category", "value": "other", "branch": "hit"},
            {"category": "billing"},
            "hit",
        ),
        (
            {"when": "contains", "path": "tags", "value": "urgent", "branch": "hit"},
            {"tags": ["urgent", "billing"]},
            "hit",
        ),
        (
            {"when": "contains", "path": "text", "value": "refund", "branch": "hit"},
            {"text": "please refund me"},
            "hit",
        ),
        ({"when": "gt", "path": "score", "value": 5, "branch": "hit"}, {"score": 7}, "hit"),
        ({"when": "gt", "path": "score", "value": 5, "branch": "hit"}, {"score": 5}, "otherwise"),
        ({"when": "gte", "path": "score", "value": 5, "branch": "hit"}, {"score": 5}, "hit"),
        ({"when": "lt", "path": "score", "value": 5, "branch": "hit"}, {"score": 4.5}, "hit"),
        ({"when": "lte", "path": "score", "value": 5, "branch": "hit"}, {"score": 5}, "hit"),
        # a missing path resolves to null: never matches a comparison ...
        (
            {"when": "equals", "path": "absent", "value": "x", "branch": "hit"},
            {"category": "billing"},
            "otherwise",
        ),
        # ... but does match `empty`
        ({"when": "empty", "path": "absent", "branch": "hit"}, {"category": "billing"}, "hit"),
        # nested dot path
        (
            {"when": "equals", "path": "meta.origin", "value": "web", "branch": "hit"},
            {"meta": {"origin": "web"}},
            "hit",
        ),
        # numeric comparison against a non-number never matches
        (
            {"when": "gt", "path": "score", "value": 5, "branch": "hit"},
            {"score": "high"},
            "otherwise",
        ),
    ],
)
def test_route_conditions(rule: dict[str, Any], value: Any, expected: str) -> None:
    assert _route(_config([rule]), value) == expected


def test_route_first_matching_rule_wins() -> None:
    config = _config(
        [
            {"when": "gt", "path": "score", "value": 8, "branch": "critical"},
            {"when": "gt", "path": "score", "value": 5, "branch": "high"},
            {"when": "not_empty", "branch": "any"},
        ]
    )
    assert _route(config, {"score": 9}) == "critical"
    assert _route(config, {"score": 6}) == "high"
    assert _route(config, {"score": 1}) == "any"
    assert _route(config, {}) == "otherwise"


def test_route_equals_on_whole_text_input() -> None:
    config = _config([{"when": "equals", "value": "yes", "branch": "hit"}], input_type="text")
    assert _route(config, "yes") == "hit"
    assert _route(config, "no") == "otherwise"


def _structured_flow() -> FlowDefinition:
    """start.payload (json) -> router with path rules -> per-branch templates."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "triage-demo",
        "expose": {"kind": "a2a"},
        "nodes": [
            {
                "id": "start_1",
                "type": "start",
                "version": 1,
                "config": {
                    "input_schema": {
                        "type": "object",
                        "properties": {"payload": {"type": "object"}},
                        "required": ["payload"],
                    }
                },
            },
            {
                "id": "route_1",
                "type": "router",
                "version": 1,
                "config": {
                    "input_type": "json",
                    "rules": [
                        {
                            "when": "equals",
                            "path": "category",
                            "value": "billing",
                            "branch": "billing",
                        }
                    ],
                    "default_branch": "other",
                },
            },
            {
                "id": "billing_1",
                "type": "template",
                "version": 1,
                "config": {"text": "billing branch"},
            },
            {
                "id": "other_1",
                "type": "template",
                "version": 1,
                "config": {"text": "other branch"},
            },
            {"id": "end_1", "type": "end", "version": 1, "config": {"output_from": ""}},
        ],
        "edges": [
            {"from": "start_1.payload", "to": "route_1.input"},
            {"from": "route_1.billing", "to": "billing_1.trigger"},
            {"from": "route_1.other", "to": "other_1.trigger"},
            {"from": "billing_1.text", "to": "end_1.input"},
            {"from": "other_1.text", "to": "end_1.input"},
        ],
    }
    return FlowDefinition.model_validate(data)


async def test_flow_routes_on_structured_output_field() -> None:
    context = ExecutionContext(
        resources=cast(ResourceService, object()), settings=RuntimeSettings(_env_file=None)
    )
    runner = FlowRunner(_structured_flow(), context)
    assert await runner.execute({"payload": {"category": "billing"}}) == "billing branch"
    assert await runner.execute({"payload": {"category": "technical"}}) == "other branch"
