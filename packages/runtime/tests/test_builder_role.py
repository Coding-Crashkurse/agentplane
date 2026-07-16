"""Builder-role gating on runtime writes (SPEC §7.1).

With ``auth_mode=oidc`` every write on the definitions and resources APIs needs
the builder role (or admin); reads and validate stay role-free. A missing role
is a 403, not a 404 — it is not an existence question. ``auth_mode=none`` is a
no-op (the plain API tests cover writes without any token).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager

from agentplane_runtime.app import create_app

from .conftest import (
    AUDIENCE,
    ISSUER,
    auth_header,
    llm_resource_body,
    load_example,
    make_settings,
    make_token,
    mock_issuer,
)


def _definition_body() -> dict[str, Any]:
    return load_example("echo-agent.yaml").canonical_dict()


async def _seed(client: httpx.AsyncClient, token: str) -> None:
    assert (
        await client.post("/api/v1/resources", json=llm_resource_body(), headers=auth_header(token))
    ).status_code == 201
    assert (
        await client.post(
            "/api/v1/definitions", json=_definition_body(), headers=auth_header(token)
        )
    ).status_code == 201


@respx.mock
async def test_user_without_builder_role_gets_403_on_every_write(
    oidc_client: httpx.AsyncClient,
) -> None:
    mock_issuer()
    builder = make_token("alice", ["user", "builder"])
    plain = make_token("bob", ["user"])
    await _seed(oidc_client, builder)

    defn = _definition_body()
    writes = [
        oidc_client.post("/api/v1/definitions", json=defn, headers=auth_header(plain)),
        oidc_client.put("/api/v1/definitions/echo-agent", json=defn, headers=auth_header(plain)),
        oidc_client.post("/api/v1/definitions/echo-agent/deploy", headers=auth_header(plain)),
        oidc_client.post(
            "/api/v1/definitions/echo-agent/deploy",
            params={"ephemeral": "true"},
            headers=auth_header(plain),
        ),
        oidc_client.post("/api/v1/definitions/echo-agent/undeploy", headers=auth_header(plain)),
        oidc_client.delete("/api/v1/definitions/echo-agent", headers=auth_header(plain)),
        oidc_client.post("/api/v1/resources", json=llm_resource_body(), headers=auth_header(plain)),
        oidc_client.put(
            "/api/v1/resources/default-llm", json=llm_resource_body(), headers=auth_header(plain)
        ),
        oidc_client.delete("/api/v1/resources/default-llm", headers=auth_header(plain)),
    ]
    for request in writes:
        response = await request
        assert response.status_code == 403, response.text
        assert "builder" in response.json()["detail"]


@respx.mock
async def test_reads_and_validate_stay_role_free(oidc_client: httpx.AsyncClient) -> None:
    mock_issuer()
    builder = make_token("alice", ["user", "builder"])
    plain = make_token("bob", ["user"])
    await _seed(oidc_client, builder)

    assert (
        await oidc_client.get("/api/v1/definitions", headers=auth_header(plain))
    ).status_code == 200
    assert (
        await oidc_client.get("/api/v1/resources", headers=auth_header(plain))
    ).status_code == 200
    validated = await oidc_client.post(
        "/api/v1/definitions/validate", json=_definition_body(), headers=auth_header(plain)
    )
    assert validated.status_code == 200
    assert validated.json()["valid"] is True


@respx.mock
async def test_builder_role_allows_the_full_write_lifecycle(
    oidc_client: httpx.AsyncClient,
) -> None:
    mock_issuer()
    builder = make_token("alice", ["user", "builder"])
    await _seed(oidc_client, builder)

    assert (
        await oidc_client.put(
            "/api/v1/definitions/echo-agent", json=_definition_body(), headers=auth_header(builder)
        )
    ).status_code == 200
    deployed = await oidc_client.post(
        "/api/v1/definitions/echo-agent/deploy", headers=auth_header(builder)
    )
    assert deployed.status_code == 200, deployed.text
    assert (
        await oidc_client.post(
            "/api/v1/definitions/echo-agent/undeploy", headers=auth_header(builder)
        )
    ).status_code == 204
    assert (
        await oidc_client.delete("/api/v1/definitions/echo-agent", headers=auth_header(builder))
    ).status_code == 204


@respx.mock
async def test_admin_without_builder_role_may_write(oidc_client: httpx.AsyncClient) -> None:
    mock_issuer()
    admin = make_token("root", ["admin"])
    await _seed(oidc_client, admin)
    assert (
        await oidc_client.delete("/api/v1/definitions/echo-agent", headers=auth_header(admin))
    ).status_code == 204


async def test_auth_off_needs_no_role(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/resources", json=llm_resource_body())).status_code == 201
    assert (await client.post("/api/v1/definitions", json=_definition_body())).status_code == 201


@pytest.fixture
async def custom_role_client() -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(
        auth_mode="oidc",
        oidc_issuer=ISSUER,
        oidc_audience=AUDIENCE,
        builder_role="publisher",
    )
    application = create_app(settings)
    async with LifespanManager(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://runtime.test"
        ) as http_client:
            yield http_client


@respx.mock
async def test_builder_role_name_is_configurable(custom_role_client: httpx.AsyncClient) -> None:
    mock_issuer()
    publisher = make_token("alice", ["user", "publisher"])
    default_builder = make_token("bob", ["user", "builder"])

    denied = await custom_role_client.post(
        "/api/v1/resources", json=llm_resource_body(), headers=auth_header(default_builder)
    )
    assert denied.status_code == 403
    assert "publisher" in denied.json()["detail"]
    assert (
        await custom_role_client.post(
            "/api/v1/resources", json=llm_resource_body(), headers=auth_header(publisher)
        )
    ).status_code == 201
