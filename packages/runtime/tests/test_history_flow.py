"""Conversation history for LLM nodes (`history: true`, SPEC §6.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from agentplane_core import FlowDefinition, ModelProviderResource
from agentplane_runtime.db import Database
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider
from agentplane_runtime.serving import EndpointManager

from .conftest import LLM_BASE, SSE_PONG, load_example, make_settings


def _history_defn() -> FlowDefinition:
    raw = load_example("echo-agent.yaml").model_dump(mode="json")
    raw["name"] = "chatty"
    for node in raw["nodes"]:
        if node["type"] == "llm_call":
            node["config"]["history"] = True
            node["config"]["history_max_turns"] = 5
    return FlowDefinition.model_validate(raw)


def _rpc_send(text: str, message_id: str, context_id: str | None = None) -> dict[str, object]:
    message: dict[str, object] = {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if context_id:
        message["contextId"] = context_id
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "method": "SendMessage",
        "params": {"message": message},
    }


@pytest.fixture
async def manager(tmp_path: Path) -> EndpointManager:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'rt.db'}")
    await db.create_all()
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    await resources.create(
        ModelProviderResource(name="default-llm", base_url=LLM_BASE, default_model="gpt-5-mini"),
        "anonymous",
    )
    return EndpointManager(resources, make_settings(task_store="database"), engine=db.engine)


@respx.mock
async def test_second_message_carries_the_first_turn(manager: EndpointManager) -> None:
    llm_bodies: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        llm_bodies.append(json.loads(request.content))
        return httpx.Response(200, text=SSE_PONG, headers={"content-type": "text/event-stream"})

    respx.post(f"{LLM_BASE}/chat/completions").mock(side_effect=record)
    respx.route(host="rt.test").pass_through()

    await manager.start(_history_defn(), 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        headers = {"A2A-Version": "1.0"}
        first = await client.post(
            "/chatty/", json=_rpc_send("mein name ist ada", "m1"), headers=headers
        )
        context_id = first.json()["result"]["task"]["contextId"]
        await client.post(
            "/chatty/", json=_rpc_send("wie heisse ich?", "m2", context_id), headers=headers
        )
    await manager.stop_all()

    assert len(llm_bodies) == 2
    # First call: no history — just system + current prompt.
    first_roles = [m["role"] for m in llm_bodies[0]["messages"]]
    assert first_roles == ["system", "user"]
    # Second call: prior user turn and the assistant reply precede the prompt.
    second = llm_bodies[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[1]["content"] == "mein name ist ada"
    assert second[3]["content"] == "wie heisse ich?"


@respx.mock
async def test_history_off_keeps_prompt_only(manager: EndpointManager) -> None:
    llm_bodies: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        llm_bodies.append(json.loads(request.content))
        return httpx.Response(200, text=SSE_PONG, headers={"content-type": "text/event-stream"})

    respx.post(f"{LLM_BASE}/chat/completions").mock(side_effect=record)
    respx.route(host="rt.test").pass_through()

    await manager.start(load_example("echo-agent.yaml"), 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        headers = {"A2A-Version": "1.0"}
        first = await client.post("/echo-agent/", json=_rpc_send("eins", "m1"), headers=headers)
        context_id = first.json()["result"]["task"]["contextId"]
        await client.post("/echo-agent/", json=_rpc_send("zwei", "m2", context_id), headers=headers)
    await manager.stop_all()

    assert [m["role"] for m in llm_bodies[1]["messages"]] == ["system", "user"]
