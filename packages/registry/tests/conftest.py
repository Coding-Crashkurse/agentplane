from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from agentplane_registry.app import create_app
from agentplane_registry.settings import RegistrySettings

AGENT_CARD_JSON: dict[str, Any] = {
    "name": "echo-agent",
    "description": "Echoes support questions",
    "version": "1",
    "supportedInterfaces": [
        {
            "url": "https://api.example/a2a/echo-agent",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ],
    "capabilities": {"streaming": True},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [{"id": "echo", "name": "echo", "description": "answer questions", "tags": ["demo"]}],
}

TOOL_CARD_JSON: dict[str, Any] = {
    "name": "support-rag",
    "description": "Search the support knowledge base",
    "url": "https://api.example/mcp/support-rag",
    "tools": [{"name": "search_support_kb", "description": "Search the KB"}],
}


def agent_entry_body(name: str = "echo-agent") -> dict[str, Any]:
    card = {**AGENT_CARD_JSON, "name": name}
    return {
        "kind": "agent",
        "card": card,
        "url": f"https://api.example/a2a/{name}",
        "tags": ["demo", "echo"],
    }


def mcp_entry_body(name: str = "support-rag") -> dict[str, Any]:
    card = {**TOOL_CARD_JSON, "name": name}
    return {
        "kind": "mcp_server",
        "card": card,
        "url": f"https://api.example/mcp/{name}",
        "tags": ["rag"],
    }


def make_settings(**overrides: Any) -> RegistrySettings:
    defaults: dict[str, Any] = {"db_url": "sqlite+aiosqlite://", "auth_mode": "none"}
    defaults.update(overrides)
    return RegistrySettings(**defaults)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(make_settings(), run_health_job=False)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://registry.test"
        ) as http_client:
            http_client.app = app  # type: ignore[attr-defined]  # handy for tests
            yield http_client
