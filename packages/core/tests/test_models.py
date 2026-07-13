"""Registry entry card handling and resource secret redaction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from a2a.types import AgentCard

from agentplane_core import (
    SECRET_PLACEHOLDER,
    ModelProviderResource,
    RegistryEntry,
    RegistryEntryCreate,
    ToolCard,
    ToolCardTool,
    VectorDBResource,
    agent_card_from_dict,
    agent_card_to_json_dict,
)

AGENT_CARD_JSON: dict[str, Any] = {
    "name": "echo-agent",
    "description": "Echoes things",
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
    "skills": [{"id": "echo", "name": "echo", "description": "echo", "tags": ["demo"]}],
}


def test_agent_card_json_roundtrip() -> None:
    card = agent_card_from_dict(dict(AGENT_CARD_JSON))
    assert isinstance(card, AgentCard)
    assert card.name == "echo-agent"
    assert agent_card_to_json_dict(card) == AGENT_CARD_JSON


def test_registry_entry_parses_agent_card_from_json() -> None:
    entry = RegistryEntry.model_validate(
        {
            "id": str(uuid4()),
            "kind": "agent",
            "card": AGENT_CARD_JSON,
            "url": "https://api.example/a2a/echo-agent",
            "tags": ["demo"],
            "owner": "user-1",
            "status": "starting",
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
    )
    assert isinstance(entry.card, AgentCard)
    assert entry.card_name == "echo-agent"
    dumped = entry.model_dump(mode="json")
    assert dumped["card"] == AGENT_CARD_JSON


def test_registry_entry_create_with_tool_card() -> None:
    create = RegistryEntryCreate(
        kind="mcp_server",
        card=ToolCard(
            name="support-rag",
            description="Search the KB",
            url="https://api.example/mcp/support-rag",
            tools=[ToolCardTool(name="search_support_kb", description="Search")],
        ),
        url="https://api.example/mcp/support-rag",
    )
    dumped = create.model_dump(mode="json")
    assert dumped["card"]["tools"][0]["name"] == "search_support_kb"
    reparsed = RegistryEntryCreate.model_validate(dumped)
    assert isinstance(reparsed.card, ToolCard)


def test_model_provider_secret_is_write_only() -> None:
    resource = ModelProviderResource(
        name="default-llm", api_key_secret="sk-verysecretvalue123456", default_model="gpt-5-mini"
    )
    dumped = resource.model_dump(mode="json")
    assert dumped["api_key_secret"] == SECRET_PLACEHOLDER
    assert "sk-verysecret" not in dumped["api_key_secret"]
    assert resource.api_key_secret == "sk-verysecretvalue123456"  # value stays readable in-process


def test_vector_db_secrets_are_write_only() -> None:
    resource = VectorDBResource.model_validate(
        {
            "kind": "qdrant",
            "name": "kb-support",
            "url": "http://qdrant:6333",
            "api_key_secret": "qdrant-key-123",
            "embedding": {
                "resource": "default-llm",
                "model": "text-embedding-3-small",
                "dimension": 1536,
            },
        }
    )
    dumped = resource.model_dump(mode="json")
    assert dumped["api_key_secret"] == SECRET_PLACEHOLDER
    assert dumped["dsn_secret"] is None
