"""Orchestrator (v1.1): registry-resolved sub-agents called over A2A.

The sub-agent is a REAL served endpoint (EndpointManager + a2a-sdk app); the
respx route forwards the orchestrator's A2A call into that ASGI app, so the
test exercises the full client -> JSON-RPC -> executor -> flow round trip.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from agentplane_core import AgentNode, FlowDefinition, ModelProviderResource
from agentplane_runtime.db import Database
from agentplane_runtime.directory import (
    AgentDirectory,
    AgentNotFoundError,
    AgentResolutionError,
    ResolvedAgent,
)
from agentplane_runtime.engine import ExecutionContext, FlowRunner, caller_token_var
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider
from agentplane_runtime.serving import EndpointManager
from agentplane_runtime.validation import validate_full

from .conftest import LLM_BASE, REGISTRY_BASE, SSE_PONG, load_example, make_settings

_CHAT = f"{LLM_BASE}/chat/completions"
_SEARCH = f"{REGISTRY_BASE}/api/v1/agents/search"
SUB_BASE = "http://sub.test"


def _entry(
    name: str, *, enabled: bool = True, url: str = "", status: str = "healthy"
) -> dict[str, Any]:
    return {
        "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "kind": "agent",
        "card": {
            "name": name,
            "description": "Echoes politely.",
            "version": "1",
            "supportedInterfaces": [
                {"url": url or f"{SUB_BASE}/{name}", "protocolBinding": "JSONRPC"}
            ],
            "capabilities": {"streaming": True},
            "skills": [
                {
                    "id": name,
                    "name": name,
                    "description": "",
                    "tags": ["flow"],
                    "examples": ["Say hi"],
                }
            ],
        },
        "url": url or f"{SUB_BASE}/{name}",
        "tags": [],
        "owner": "anonymous",
        "group": "",
        "status": status,
        "enabled": enabled,
        "created_at": "2026-07-19T00:00:00Z",
        "updated_at": "2026-07-19T00:00:00Z",
    }


def _page(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"items": items, "total": len(items), "limit": 200, "offset": 0}


@respx.mock
async def test_directory_resolves_and_caches() -> None:
    directory = AgentDirectory(make_settings(registry_url=REGISTRY_BASE))
    route = respx.get(_SEARCH).mock(
        return_value=httpx.Response(200, json=_page([_entry("echo-agent")]))
    )
    resolved = await directory.resolve("echo-agent")
    assert resolved.url == f"{SUB_BASE}/echo-agent"
    assert resolved.description == "Echoes politely."
    assert resolved.examples == ("Say hi",)
    again = await directory.resolve("echo-agent")
    assert again == resolved
    assert route.call_count == 1  # second hit came from the cache


@respx.mock
async def test_directory_unknown_agent_raises() -> None:
    directory = AgentDirectory(make_settings(registry_url=REGISTRY_BASE))
    respx.get(_SEARCH).mock(return_value=httpx.Response(200, json=_page([])))
    with pytest.raises(AgentNotFoundError):
        await directory.resolve("ghost")


@respx.mock
async def test_directory_disabled_agent_is_not_resolved() -> None:
    directory = AgentDirectory(make_settings(registry_url=REGISTRY_BASE))
    respx.get(_SEARCH).mock(
        return_value=httpx.Response(200, json=_page([_entry("echo-agent", enabled=False)]))
    )
    with pytest.raises(AgentNotFoundError):
        await directory.resolve("echo-agent")


async def test_directory_without_registry_raises() -> None:
    directory = AgentDirectory(make_settings(registry_url=""))
    with pytest.raises(AgentResolutionError):
        await directory.resolve("echo-agent")


def _orchestrator_defn(sub_name: str = "echo-agent") -> FlowDefinition:
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "orchestrator-demo",
        "expose": {"kind": "a2a"},
        "nodes": [
            {
                "id": "start_1",
                "type": "start",
                "version": 1,
                "config": {
                    "input_schema": {
                        "type": "object",
                        "properties": {"request": {"type": "string"}},
                        "required": ["request"],
                    }
                },
            },
            {
                "id": "agent_1",
                "type": "agent",
                "version": 1,
                "config": {
                    "resource": "default-llm",
                    "prompt": "{request}",
                    "agents": [{"name": sub_name}],
                    "max_iterations": 3,
                },
            },
            {"id": "end_1", "type": "end", "version": 1, "config": {"output_from": "agent_1.text"}},
        ],
        "edges": [
            {"from": "start_1.request", "to": "agent_1.request"},
            {"from": "agent_1.text", "to": "end_1.input"},
        ],
    }
    return FlowDefinition.model_validate(data)


async def _resources() -> ResourceService:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    await resources.create(
        ModelProviderResource(name="default-llm", base_url=LLM_BASE, default_model="gpt-5-mini"),
        "anonymous",
    )
    return resources


def _delegate_turn(tool_name: str, message: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": f'{{"message": "{message}"}}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )


def _final(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def _forward_to(app: object) -> Any:
    """respx side effect forwarding a request into an ASGI app (the sub-agent)."""
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]

    async def forward(request: httpx.Request) -> httpx.Response:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() in ("authorization", "content-type", "a2a-version")
        }
        async with httpx.AsyncClient(transport=transport, base_url=SUB_BASE) as fwd:
            upstream = await fwd.request(
                request.method, request.url.path, content=request.content, headers=headers
            )
        return httpx.Response(
            upstream.status_code,
            content=upstream.content,
            headers={"content-type": upstream.headers.get("content-type", "application/json")},
        )

    return forward


@respx.mock
async def test_orchestrator_delegates_to_sub_agent_over_a2a() -> None:
    resources = await _resources()
    settings = make_settings()
    sub_manager = EndpointManager(resources, settings)
    await sub_manager.start(load_example("echo-agent.yaml"), 1)

    directory = AgentDirectory(make_settings(registry_url=REGISTRY_BASE))
    respx.get(_SEARCH).mock(return_value=httpx.Response(200, json=_page([_entry("echo-agent")])))
    sub_route = respx.post(f"{SUB_BASE}/echo-agent/").mock(side_effect=_forward_to(sub_manager.a2a))
    # 1st: orchestrator delegates; 2nd: sub-agent's own LLM call (streams);
    # 3rd: orchestrator's final answer.
    respx.post(_CHAT).mock(
        side_effect=[
            _delegate_turn("agent-echo-agent", "ping"),
            httpx.Response(200, text=SSE_PONG, headers={"content-type": "text/event-stream"}),
            _final("The echo agent said: pong!"),
        ]
    )

    defn = _orchestrator_defn()
    runner = FlowRunner(
        defn,
        ExecutionContext(
            resources=resources,
            settings=settings,
            flow_name=defn.name,
            flow_version=1,
            agents=directory,
        ),
    )
    token = caller_token_var.set("caller-jwt-123")
    try:
        result = await runner.execute({"request": "say pong"})
    finally:
        caller_token_var.reset(token)

    assert result == "The echo agent said: pong!"
    assert sub_route.call_count == 1
    sent = sub_route.calls[0].request
    assert sent.headers["Authorization"] == "Bearer caller-jwt-123"
    body = sent.read().decode()
    assert '"agentplane_call_depth"' in body  # depth metadata travels with the hop

    await sub_manager.stop_all()


@respx.mock
async def test_orchestrator_depth_limit_blocks_outgoing_call() -> None:
    resources = await _resources()
    settings = make_settings(max_agent_call_depth=2)
    directory = AgentDirectory(make_settings(registry_url=REGISTRY_BASE))
    respx.get(_SEARCH).mock(return_value=httpx.Response(200, json=_page([_entry("echo-agent")])))
    sub_route = respx.post(f"{SUB_BASE}/echo-agent/").mock(return_value=httpx.Response(500))
    chat_route = respx.post(_CHAT).mock(
        side_effect=[_delegate_turn("agent-echo-agent", "ping"), _final("gave up")]
    )

    defn = _orchestrator_defn()
    runner = FlowRunner(
        defn,
        ExecutionContext(
            resources=resources, settings=settings, flow_name=defn.name, agents=directory
        ),
    )
    # already at the limit: the outgoing call would be depth 3 > 2
    result = await runner.execute({"request": "loop"}, call_depth=2)
    assert result == "gave up"
    assert sub_route.call_count == 0  # blocked sender-side, no doomed call
    assert chat_route.call_count == 2


@respx.mock
async def test_serving_rejects_too_deep_a2a_requests() -> None:
    resources = await _resources()
    manager = EndpointManager(resources, make_settings(max_agent_call_depth=3))
    await manager.start(load_example("echo-agent.yaml"), 1)
    respx.route(host="rt.test").pass_through()

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": "m1",
                "role": "ROLE_USER",
                "parts": [{"text": "ping"}],
                "metadata": {"agentplane_call_depth": 99},
            }
        },
    }
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        response = await client.post("/echo-agent/", json=payload, headers={"A2A-Version": "1.0"})
    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    assert "depth" in str(task["status"])
    await manager.stop_all()


async def test_validate_e061_without_registry() -> None:
    resources = await _resources()
    result = await validate_full(_orchestrator_defn(), resources, None)
    assert not result.valid
    assert ("E061", "nodes/agent_1/config/agents/0/name") in [
        (i.code, i.path) for i in result.issues
    ]


@respx.mock
async def test_validate_w004_unhealthy_agent_is_a_warning() -> None:
    resources = await _resources()
    directory = AgentDirectory(make_settings(registry_url=REGISTRY_BASE))
    respx.get(_SEARCH).mock(
        return_value=httpx.Response(200, json=_page([_entry("echo-agent", status="unhealthy")]))
    )
    result = await validate_full(_orchestrator_defn(), resources, directory)
    assert result.valid  # a warning, not an error — the agent may recover
    assert ("W004", "nodes/agent_1/config/agents/0/name") in [
        (i.code, i.path) for i in result.issues
    ]


@respx.mock
async def test_validate_e062_unknown_agent() -> None:
    resources = await _resources()
    directory = AgentDirectory(make_settings(registry_url=REGISTRY_BASE))
    respx.get(_SEARCH).mock(return_value=httpx.Response(200, json=_page([])))
    result = await validate_full(_orchestrator_defn("ghost-agent"), resources, directory)
    assert not result.valid
    assert ("E062", "nodes/agent_1/config/agents/0/name") in [
        (i.code, i.path) for i in result.issues
    ]


@respx.mock
async def test_agent_tools_include_sub_agent_schema() -> None:
    resources = await _resources()
    directory = AgentDirectory(make_settings(registry_url=REGISTRY_BASE))
    respx.get(_SEARCH).mock(return_value=httpx.Response(200, json=_page([_entry("echo-agent")])))
    defn = _orchestrator_defn()
    node = next(n for n in defn.nodes if isinstance(n, AgentNode))
    runner = FlowRunner(
        defn,
        ExecutionContext(
            resources=resources, settings=make_settings(), flow_name=defn.name, agents=directory
        ),
    )
    schemas, targets, agent_targets = await runner._agent_tools(node.config)
    assert targets == {}
    assert list(agent_targets) == ["agent-echo-agent"]
    assert agent_targets["agent-echo-agent"] == ResolvedAgent(
        name="echo-agent",
        url=f"{SUB_BASE}/echo-agent",
        description="Echoes politely.",
        examples=("Say hi",),
        status="healthy",
    )
    function = schemas[0]["function"]
    assert isinstance(function, dict)
    assert function["name"] == "agent-echo-agent"
    assert "Echoes politely." in str(function["description"])
    assert "Say hi" in str(function["description"])
