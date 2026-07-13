from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from asgi_lifespan import LifespanManager
from cryptography.fernet import Fernet
from fastapi import FastAPI

from agentplane_core import FlowDefinition
from agentplane_runtime.app import create_app
from agentplane_runtime.settings import RuntimeSettings

EXAMPLES_DIR = Path(__file__).parents[3] / "examples"

PUBLIC_BASE = "https://api.example"
LLM_BASE = "http://llm.test/v1"
QDRANT_BASE = "http://qdrant.test"
REGISTRY_BASE = "http://registry.test"

SSE_PONG = (
    'data: {"choices":[{"delta":{"content":"po"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"ng!"}}]}\n\n'
    "data: [DONE]\n\n"
)


def load_example(name: str) -> FlowDefinition:
    with (EXAMPLES_DIR / name).open(encoding="utf-8") as fh:
        return FlowDefinition.model_validate(yaml.safe_load(fh))


def make_settings(**overrides: Any) -> RuntimeSettings:
    defaults: dict[str, Any] = {
        "db_url": "sqlite+aiosqlite://",
        "public_base_url": PUBLIC_BASE,
        "secret_key": Fernet.generate_key().decode("ascii"),
        "registry_url": "",
        "llm_base_url": "",
        "auth_mode": "none",
    }
    defaults.update(overrides)
    return RuntimeSettings(**defaults)


def llm_resource_body() -> dict[str, Any]:
    return {
        "kind": "model_provider",
        "name": "default-llm",
        "base_url": LLM_BASE,
        "api_key_secret": "sk-secret-value-123456",
        "default_model": "gpt-5-mini",
    }


def vector_db_body(dimension: int = 3) -> dict[str, Any]:
    return {
        "kind": "qdrant",
        "name": "kb-support",
        "url": QDRANT_BASE,
        "embedding": {
            "resource": "default-llm",
            "model": "text-embedding-3-small",
            "dimension": dimension,
        },
    }


@pytest.fixture
async def app() -> AsyncIterator[FastAPI]:
    application = create_app(make_settings())
    async with LifespanManager(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://runtime.test"
    ) as http_client:
        yield http_client


async def create_default_resources(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/resources", json=llm_resource_body())
    assert response.status_code == 201, response.text
    response = await client.post("/api/v1/resources", json=vector_db_body())
    assert response.status_code == 201, response.text
