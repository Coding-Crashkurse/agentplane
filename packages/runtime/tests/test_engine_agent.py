"""Agent node: the LLM + tool loop (LLM mocked via respx, MCP via monkeypatch)."""

from __future__ import annotations

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


@respx.mock
async def test_agent_answers_without_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, node = await _agent_runner()

    async def no_tools(refs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        return [], {}

    monkeypatch.setattr(runner, "_agent_tools", no_tools)
    respx.post(_CHAT).mock(return_value=_final("Paris."))
    out = await runner._run_agent(node, {"question": "capital of France?"})
    assert out["agent_1.text"] == "Paris."


@respx.mock
async def test_agent_runs_tool_then_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, node = await _agent_runner()
    calls: list[object] = []

    async def fake_tools(refs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        schema: dict[str, object] = {"type": "function", "function": {"name": "search"}}
        return [schema], {}

    async def fake_call(call: object, targets: object) -> str:
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
async def test_agent_max_iterations_forces_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, node = await _agent_runner()  # example caps max_iterations at 4

    async def fake_tools(refs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        schema: dict[str, object] = {"type": "function", "function": {"name": "search"}}
        return [schema], {}

    async def fake_call(call: object, targets: object) -> str:
        return "again"

    monkeypatch.setattr(runner, "_agent_tools", fake_tools)
    monkeypatch.setattr(runner, "_call_agent_tool", fake_call)
    # every turn asks for a tool → the loop exhausts, then one forced final answer
    respx.post(_CHAT).mock(
        side_effect=[_tool_turn(), _tool_turn(), _tool_turn(), _tool_turn(), _final("forced")]
    )
    out = await runner._run_agent(node, {"question": "loop"})
    assert out["agent_1.text"] == "forced"
