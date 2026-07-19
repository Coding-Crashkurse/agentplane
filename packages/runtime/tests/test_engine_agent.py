"""Agent node: the LLM + tool loop (LLM mocked via respx, MCP via monkeypatch)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from agentplane_core import AgentNode, ModelProviderResource
from agentplane_runtime.db import Database
from agentplane_runtime.engine import ExecutionContext, FlowRunner
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider

from .conftest import LLM_BASE, load_example, make_settings

_CHAT = f"{LLM_BASE}/chat/completions"


def _tool_turn() -> httpx.Response:
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
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
    )


def _final(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


async def _agent_runner() -> tuple[FlowRunner, AgentNode]:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    await resources.create(
        ModelProviderResource(name="default-llm", base_url=LLM_BASE, default_model="gpt-5-mini"),
        "anonymous",
    )
    defn = load_example("agent-with-tools.yaml")
    node = next(n for n in defn.nodes if isinstance(n, AgentNode))
    runner = FlowRunner(
        defn,
        ExecutionContext(
            resources=resources, settings=make_settings(), flow_name=defn.name, flow_version=1
        ),
    )
    return runner, node


_ToolsResult = tuple[list[dict[str, object]], dict[str, object], dict[str, object]]


@respx.mock
async def test_agent_answers_without_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, node = await _agent_runner()

    async def no_tools(config: object) -> _ToolsResult:
        return [], {}, {}

    monkeypatch.setattr(runner, "_agent_tools", no_tools)
    respx.post(_CHAT).mock(return_value=_final("Paris."))
    out = await runner._run_agent(node, {"question": "capital of France?"})
    assert out["agent_1.text"] == "Paris."


@respx.mock
async def test_agent_runs_tool_then_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, node = await _agent_runner()
    calls: list[object] = []

    async def fake_tools(config: object) -> _ToolsResult:
        schema: dict[str, object] = {"type": "function", "function": {"name": "search"}}
        return [schema], {}, {}

    async def fake_call(call: object, targets: object, agent_targets: object) -> str:
        calls.append(call)
        return "search result: 42"

    monkeypatch.setattr(runner, "_agent_tools", fake_tools)
    monkeypatch.setattr(runner, "_call_agent_tool", fake_call)
    route = respx.post(_CHAT).mock(side_effect=[_tool_turn(), _final("The answer is 42.")])

    out = await runner._run_agent(node, {"question": "the answer?"})
    assert out["agent_1.text"] == "The answer is 42."
    assert len(calls) == 1
    assert route.call_count == 2


@respx.mock
async def test_tool_calls_of_one_turn_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each call waits for the OTHER to start — sequential execution would deadlock."""
    runner, node = await _agent_runner()
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    async def fake_tools(config: object) -> _ToolsResult:
        schema: dict[str, object] = {"type": "function", "function": {"name": "search"}}
        return [schema], {}, {}

    async def fake_call(call: object, targets: object, agent_targets: object) -> str:
        call_id = call.get("id") if isinstance(call, dict) else ""
        if call_id == "c1":
            first_started.set()
            await asyncio.wait_for(second_started.wait(), timeout=5)
            return "one"
        second_started.set()
        await asyncio.wait_for(first_started.wait(), timeout=5)
        return "two"

    monkeypatch.setattr(runner, "_agent_tools", fake_tools)
    monkeypatch.setattr(runner, "_call_agent_tool", fake_call)

    two_calls = httpx.Response(
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
                                "function": {"name": "search", "arguments": "{}"},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "search", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ]
        },
    )
    respx.post(_CHAT).mock(side_effect=[two_calls, _final("both done")])
    out = await runner._run_agent(node, {"question": "fan out"})
    assert out["agent_1.text"] == "both done"
    assert first_started.is_set() and second_started.is_set()


@respx.mock
async def test_agent_max_iterations_forces_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, node = await _agent_runner()  # example caps max_iterations at 4

    async def fake_tools(config: object) -> _ToolsResult:
        schema: dict[str, object] = {"type": "function", "function": {"name": "search"}}
        return [schema], {}, {}

    async def fake_call(call: object, targets: object, agent_targets: object) -> str:
        return "again"

    monkeypatch.setattr(runner, "_agent_tools", fake_tools)
    monkeypatch.setattr(runner, "_call_agent_tool", fake_call)
    # every turn asks for a tool → the loop exhausts, then one forced final answer
    respx.post(_CHAT).mock(
        side_effect=[_tool_turn(), _tool_turn(), _tool_turn(), _tool_turn(), _final("forced")]
    )
    out = await runner._run_agent(node, {"question": "loop"})
    assert out["agent_1.text"] == "forced"
