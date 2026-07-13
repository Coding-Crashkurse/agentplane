"""Client behavior against a mocked HTTP API (respx)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
import respx
from httpx import Response

from agentplane_core import (
    FlowDefinition,
    ModelProviderResource,
    RegistryEntryCreate,
    ToolCard,
)
from agentplane_sdk import (
    NotFoundError,
    RegistryClient,
    RuntimeClient,
    TransportError,
    ValidationFailedError,
)

RUNTIME = "http://runtime.test"
REGISTRY = "http://registry.test"


@respx.mock
async def test_validate_returns_result(echo_definition: FlowDefinition) -> None:
    route = respx.post(f"{RUNTIME}/api/v1/definitions/validate").mock(
        return_value=Response(200, json={"valid": True, "issues": []})
    )
    async with RuntimeClient(RUNTIME) as client:
        result = await client.validate(echo_definition)
    assert result.valid
    assert route.called


@respx.mock
async def test_deploy_returns_deployment_info() -> None:
    respx.post(f"{RUNTIME}/api/v1/definitions/echo-agent/deploy").mock(
        return_value=Response(
            200,
            json={
                "name": "echo-agent",
                "version": 3,
                "endpoint_url": "https://api.example/a2a/echo-agent",
                "registry_id": str(uuid4()),
            },
        )
    )
    async with RuntimeClient(RUNTIME) as client:
        info = await client.deploy("echo-agent")
    assert info.version == 3
    assert info.endpoint_url.endswith("/a2a/echo-agent")


@respx.mock
async def test_422_maps_to_validation_failed(echo_definition: FlowDefinition) -> None:
    respx.post(f"{RUNTIME}/api/v1/definitions").mock(
        return_value=Response(
            422,
            json={
                "valid": False,
                "issues": [
                    {
                        "code": "E020",
                        "severity": "error",
                        "path": "nodes/call_1/config/resource",
                        "message": "unknown resource",
                    }
                ],
            },
        )
    )
    async with RuntimeClient(RUNTIME) as client:
        with pytest.raises(ValidationFailedError) as excinfo:
            await client.create_draft(echo_definition)
    assert excinfo.value.result.issues[0].code == "E020"


@respx.mock
async def test_404_maps_to_not_found() -> None:
    respx.get(f"{RUNTIME}/api/v1/definitions/nope").mock(return_value=Response(404, text="no"))
    async with RuntimeClient(RUNTIME) as client:
        with pytest.raises(NotFoundError):
            await client.get("nope")


async def test_connect_error_maps_to_transport_error() -> None:
    async with RuntimeClient("http://127.0.0.1:1") as client:
        with pytest.raises(TransportError):
            await client.get("x")


@respx.mock
async def test_create_resource_sends_real_secret() -> None:
    route = respx.post(f"{RUNTIME}/api/v1/resources").mock(
        return_value=Response(
            201,
            json={
                "kind": "model_provider",
                "name": "default-llm",
                "api_key_secret": "•••",
                "default_model": "gpt-5-mini",
            },
        )
    )
    resource = ModelProviderResource(name="default-llm", api_key_secret="sk-real-value-123456")
    async with RuntimeClient(RUNTIME) as client:
        created = await client.create_resource(resource)
    sent = route.calls.last.request.content.decode()
    assert "sk-real-value-123456" in sent  # create requests must transmit the secret
    assert isinstance(created, ModelProviderResource)
    assert created.name == "default-llm"
    assert created.api_key_secret == "•••"  # responses stay redacted


@respx.mock
async def test_registry_register_and_search() -> None:
    entry_json = {
        "id": str(uuid4()),
        "kind": "mcp_server",
        "card": {
            "name": "support-rag",
            "description": "kb",
            "url": "https://gw/mcp/x",
            "tools": [],
        },
        "url": "https://gw/mcp/x",
        "tags": ["rag"],
        "owner": "anonymous",
        "status": "starting",
        "last_seen": None,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    respx.post(f"{REGISTRY}/api/v1/agents").mock(return_value=Response(201, json=entry_json))
    search_route = respx.get(f"{REGISTRY}/api/v1/agents/search").mock(
        return_value=Response(
            200, json={"items": [entry_json], "total": 1, "limit": 50, "offset": 0}
        )
    )
    async with RegistryClient(REGISTRY, token="secret-token") as client:
        created = await client.register(
            RegistryEntryCreate(
                kind="mcp_server",
                card=ToolCard(name="support-rag", url="https://gw/mcp/x"),
                url="https://gw/mcp/x",
                tags=["rag"],
            )
        )
        page = await client.search("support", tags=["rag"], semantic=True)
    assert created.kind == "mcp_server"
    assert page.total == 1
    request = search_route.calls.last.request
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert "semantic=true" in str(request.url)
