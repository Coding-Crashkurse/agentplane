"""A2A + MCP endpoint serving (SPEC §6.4/§6.5)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from cryptography.fernet import Fernet
from fastmcp import Client

from agentplane_core import ModelProviderResource, VectorDBResource
from agentplane_runtime.db import Database
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider
from agentplane_runtime.serving import (
    EndpointManager,
    bind_message_to_inputs,
    build_agent_card,
    build_mcp_server,
)
from agentplane_runtime.settings import RuntimeSettings

from .conftest import LLM_BASE, QDRANT_BASE, SSE_PONG, load_example, make_settings, vector_db_body


@pytest.fixture
async def manager() -> AsyncIterator[EndpointManager]:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    await resources.create(
        ModelProviderResource(name="default-llm", base_url=LLM_BASE, default_model="gpt-5-mini"),
        "anonymous",
    )
    await resources.create(VectorDBResource.model_validate(vector_db_body()), "anonymous")
    settings: RuntimeSettings = make_settings()
    endpoint_manager = EndpointManager(resources, settings)
    yield endpoint_manager
    await endpoint_manager.stop_all()


def _rpc_send(text: str) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": text}]}},
    }


async def test_agent_card_is_served(manager: EndpointManager) -> None:
    defn = load_example("echo-agent.yaml")
    await manager.start(defn, 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        card = (await client.get("/echo-agent/.well-known/agent-card.json")).json()
    assert card["name"] == "echo-agent"
    assert card["supportedInterfaces"][0]["url"] == "https://api.example/a2a/echo-agent"
    assert card["capabilities"]["streaming"] is True


@respx.mock
async def test_a2a_send_message_returns_flow_output(manager: EndpointManager) -> None:
    respx.post(f"{LLM_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, text=SSE_PONG, headers={"content-type": "text/event-stream"}
        )
    )
    respx.route(host="rt.test").pass_through()
    defn = load_example("echo-agent.yaml")
    await manager.start(defn, 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        response = await client.post(
            "/echo-agent/", json=_rpc_send("ping"), headers={"A2A-Version": "1.0"}
        )
    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["text"] == "pong!"


@respx.mock
async def test_a2a_flow_failure_becomes_task_failure(manager: EndpointManager) -> None:
    respx.post(f"{LLM_BASE}/chat/completions").mock(return_value=httpx.Response(500))
    respx.route(host="rt.test").pass_through()
    defn = load_example("echo-agent.yaml")
    await manager.start(defn, 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        response = await client.post(
            "/echo-agent/", json=_rpc_send("ping"), headers={"A2A-Version": "1.0"}
        )
    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"


async def test_unknown_endpoint_is_404(manager: EndpointManager) -> None:
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        assert (await client.get("/ghost/.well-known/agent-card.json")).status_code == 404


async def test_stop_unmounts_endpoint(manager: EndpointManager) -> None:
    defn = load_example("echo-agent.yaml")
    await manager.start(defn, 1)
    assert manager.a2a.mounted() == ["echo-agent"]
    await manager.stop("echo-agent")
    assert manager.a2a.mounted() == []


@respx.mock
async def test_mcp_tool_runs_rag_flow(manager: EndpointManager) -> None:
    respx.post(f"{LLM_BASE}/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    respx.post(f"{QDRANT_BASE}/collections/support_docs/points/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "result": [
                    {"score": 0.9, "payload": {"text": "Reset the router.", "source": "kb/1"}}
                ]
            },
        )
    )
    llm_route = respx.post(f"{LLM_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "Please reset the router."}}]}
        )
    )
    defn = load_example("support-rag.yaml")
    server = build_mcp_server(defn, manager._runner_factory(defn, 1))
    async with Client(server) as client:
        tools = await client.list_tools()
        assert [t.name for t in tools] == ["search_support_kb"]
        schema = tools[0].inputSchema
        assert schema["properties"] == {"query": {"type": "string"}}
        result = await client.call_tool("search_support_kb", {"query": "wifi broken"})
    text = result.content[0].text if result.content else ""
    assert text == "Please reset the router."
    # the retrieved documents were rendered into the prompt
    request_body = llm_route.calls.last.request.content.decode()
    assert "Reset the router." in request_body
    assert "wifi broken" in request_body


async def test_mcp_endpoint_mounts_and_unmounts(manager: EndpointManager) -> None:
    defn = load_example("support-rag.yaml")
    endpoint = await manager.start(defn, 1)
    assert endpoint.public_url == "https://api.example/mcp/support-rag"
    assert manager.mcp.mounted() == ["support-rag"]
    await manager.stop("support-rag")
    assert manager.mcp.mounted() == []


def test_bind_message_single_string_input() -> None:
    defn = load_example("echo-agent.yaml")
    assert bind_message_to_inputs(defn, "hello") == {"message": "hello"}


def test_bind_message_json_object_for_multi_input() -> None:
    defn = load_example("echo-agent.yaml")
    start = defn.nodes[2]
    assert start.type == "start"
    multi = defn.model_copy(deep=True)
    schema = multi.node_map()["start_1"].config.input_schema  # type: ignore[union-attr]
    schema["properties"]["extra"] = {"type": "string"}
    schema["required"] = ["message", "extra"]
    assert bind_message_to_inputs(multi, '{"message": "m", "extra": "e"}') == {
        "message": "m",
        "extra": "e",
    }
    with pytest.raises(ValueError, match="JSON object"):
        bind_message_to_inputs(multi, "not json")


def test_agent_card_fields_derive_from_definition() -> None:
    defn = load_example("support-rag.yaml")
    card = build_agent_card(defn, "https://api.example/a2a/support-rag")
    assert card.name == "support-rag"
    assert card.skills[0].tags == ["support", "rag"]
    assert card.supported_interfaces[0].protocol_version == "1.0"


def test_agent_card_carries_expose_examples() -> None:
    """expose.examples land on the skill so calling agents can route (FEEDBACK 2.2)."""
    defn = load_example("echo-agent.yaml")
    card = build_agent_card(defn, "https://api.example/a2a/echo-agent")
    assert list(card.skills[0].examples) == [
        "What is the capital of France?",
        "Summarise this paragraph in one sentence.",
    ]


def test_agent_card_version_defaults_to_the_deploy_counter() -> None:
    defn = load_example("echo-agent.yaml")
    assert build_agent_card(defn, "https://api.example/a2a/echo-agent").version == "1"
    labelled = build_agent_card(defn, "https://api.example/a2a/echo-agent", "2.1.0")
    assert labelled.version == "2.1.0"


async def test_browser_get_on_the_endpoint_explains_itself(manager: EndpointManager) -> None:
    """The JSON-RPC binding is POST-only; a plain GET used to be a bare 405 (FEEDBACK 2.4)."""
    defn = load_example("echo-agent.yaml")
    await manager.start(defn, 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        response = await client.get("/echo-agent/")
    assert response.status_code == 200
    body = response.json()
    assert body["protocol"] == "A2A"
    assert body["agent_card_url"] == (
        "https://api.example/a2a/echo-agent/.well-known/agent-card.json"
    )
    assert "A2A-Version: 1.0" in body["hint"]
