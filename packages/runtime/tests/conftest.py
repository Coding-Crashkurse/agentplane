from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
import respx
import yaml
from asgi_lifespan import LifespanManager
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa
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


ISSUER = "http://keycloak.test/realms/agentplane"
AUDIENCE = "agentplane"
KID = "test-key"

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks() -> dict[str, Any]:
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(_PRIVATE_KEY.public_key(), as_dict=True)
    return {"keys": [{**public_jwk, "kid": KID, "alg": "RS256", "use": "sig"}]}


def make_token(sub: str, roles: list[str], groups: list[str] | None = None) -> str:
    claims = {
        "sub": sub,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 600,
        "realm_access": {"roles": roles},
        "groups": groups or [],
    }
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256", headers={"kid": KID})


def mock_issuer() -> None:
    respx.get(f"{ISSUER}/.well-known/openid-configuration").mock(
        return_value=httpx.Response(200, json={"jwks_uri": f"{ISSUER}/jwks"})
    )
    respx.get(f"{ISSUER}/jwks").mock(return_value=httpx.Response(200, json=_jwks()))


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def oidc_client() -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(auth_mode="oidc", oidc_issuer=ISSUER, oidc_audience=AUDIENCE)
    application = create_app(settings)
    async with LifespanManager(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://runtime.test"
        ) as http_client:
            yield http_client
