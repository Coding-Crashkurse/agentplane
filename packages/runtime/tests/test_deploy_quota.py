"""Per-owner deployment quota (SPEC §7.2).

A SaaS guard-rail caps how many non-ephemeral definitions one owner may keep
deployed at once (``AGENTPLANE_RUNTIME_MAX_DEPLOYMENTS_PER_OWNER``). Redeploying
an already-deployed definition consumes no new slot, undeploy frees one, admins
bypass, and the cap is off (unlimited) by default.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
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


@asynccontextmanager
async def _runtime(limit: int, **overrides: Any) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(max_deployments_per_owner=limit, **overrides)
    application = create_app(settings)
    async with LifespanManager(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://runtime.test"
        ) as http_client:
            yield http_client


def _named_echo(name: str) -> dict[str, Any]:
    body = load_example("echo-agent.yaml").canonical_dict()
    body["name"] = name
    return body


async def _create_llm(client: httpx.AsyncClient, headers: dict[str, str] | None = None) -> None:
    response = await client.post(
        "/api/v1/resources", json=llm_resource_body(), headers=headers or {}
    )
    assert response.status_code == 201, response.text


async def _deploy_new(
    client: httpx.AsyncClient, name: str, headers: dict[str, str] | None = None
) -> httpx.Response:
    """Create a fresh echo draft under ``name`` and deploy it."""
    draft = await client.post("/api/v1/definitions", json=_named_echo(name), headers=headers or {})
    assert draft.status_code == 201, draft.text
    return await client.post(f"/api/v1/definitions/{name}/deploy", headers=headers or {})


async def test_quota_blocks_deploy_over_the_limit() -> None:
    async with _runtime(2) as client:
        await _create_llm(client)
        assert (await _deploy_new(client, "echo-one")).status_code == 200
        assert (await _deploy_new(client, "echo-two")).status_code == 200

        blocked = await _deploy_new(client, "echo-three")
        assert blocked.status_code == 429
        body = blocked.json()["detail"]
        assert body["error"] == "deployment_quota_exceeded"
        assert body["limit"] == 2
        assert body["deployed"] == 2


async def test_redeploy_of_deployed_definition_does_not_count() -> None:
    async with _runtime(2) as client:
        await _create_llm(client)
        assert (await _deploy_new(client, "echo-one")).status_code == 200
        assert (await _deploy_new(client, "echo-two")).status_code == 200
        # echo-one is already deployed; redeploying it must stay allowed at the cap.
        redeploy = await client.post("/api/v1/definitions/echo-one/deploy")
        assert redeploy.status_code == 200


async def test_undeploy_frees_a_slot() -> None:
    async with _runtime(2) as client:
        await _create_llm(client)
        assert (await _deploy_new(client, "echo-one")).status_code == 200
        assert (await _deploy_new(client, "echo-two")).status_code == 200
        # echo-three's draft is created here but the deploy is refused (at limit).
        assert (await _deploy_new(client, "echo-three")).status_code == 429

        assert (await client.post("/api/v1/definitions/echo-one/undeploy")).status_code == 204
        # a slot is now free, so the previously-refused deploy goes through.
        assert (await client.post("/api/v1/definitions/echo-three/deploy")).status_code == 200


async def test_quota_is_off_by_default() -> None:
    async with _runtime(0) as client:  # 0 = unlimited (the default)
        await _create_llm(client)
        for name in ("echo-one", "echo-two", "echo-three", "echo-four", "echo-five"):
            assert (await _deploy_new(client, name)).status_code == 200


@respx.mock
async def test_admin_bypasses_the_quota() -> None:
    mock_issuer()
    respx.route(host="runtime.test").pass_through()
    async with _runtime(1, auth_mode="oidc", oidc_issuer=ISSUER, oidc_audience=AUDIENCE) as client:
        admin = auth_header(make_token("root", ["admin"]))
        await _create_llm(client, admin)
        # Limit is 1, yet the admin deploys three definitions: the cap is bypassed.
        assert (await _deploy_new(client, "echo-one", admin)).status_code == 200
        assert (await _deploy_new(client, "echo-two", admin)).status_code == 200
        assert (await _deploy_new(client, "echo-three", admin)).status_code == 200


@respx.mock
async def test_non_admin_builder_is_capped_under_oidc() -> None:
    mock_issuer()
    respx.route(host="runtime.test").pass_through()
    async with _runtime(1, auth_mode="oidc", oidc_issuer=ISSUER, oidc_audience=AUDIENCE) as client:
        builder = auth_header(make_token("alice", ["user", "builder"]))
        await _create_llm(client, builder)
        assert (await _deploy_new(client, "echo-one", builder)).status_code == 200

        blocked = await _deploy_new(client, "echo-two", builder)
        assert blocked.status_code == 429
        assert blocked.json()["detail"]["owner"] == "alice"
