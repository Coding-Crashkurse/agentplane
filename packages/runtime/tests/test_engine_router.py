"""Router/template execution (v1.1): only the chosen branch runs."""

from __future__ import annotations

from typing import Any, cast

from agentplane_core import FlowDefinition
from agentplane_runtime.engine import ExecutionContext, FlowRunner
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
