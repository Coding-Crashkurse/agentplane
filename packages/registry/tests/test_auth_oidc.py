"""OIDC auth: JWT validation, role mapping, ownership enforcement (SPEC §5.5/§7.1)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import jwt
import pytest
import respx
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric import rsa

from agentplane_registry.app import create_app

from .conftest import agent_entry_body, make_settings

ISSUER = "http://keycloak.test/realms/agentplane"
AUDIENCE = "agentplane"
KID = "test-key"

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks() -> dict[str, Any]:
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(_PRIVATE_KEY.public_key(), as_dict=True)
    return {"keys": [{**public_jwk, "kid": KID, "alg": "RS256", "use": "sig"}]}


def make_token(
    sub: str, roles: list[str], groups: list[str] | None = None, *, issuer: str = ISSUER
) -> str:
    claims = {
        "sub": sub,
        "iss": issuer,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 600,
        "realm_access": {"roles": roles},
        "groups": groups or [],
    }
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256", headers={"kid": KID})


def _mock_issuer() -> None:
    respx.get(f"{ISSUER}/.well-known/openid-configuration").mock(
        return_value=httpx.Response(200, json={"jwks_uri": f"{ISSUER}/jwks"})
    )
    respx.get(f"{ISSUER}/jwks").mock(return_value=httpx.Response(200, json=_jwks()))


@pytest.fixture
async def oidc_client() -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(auth_mode="oidc", oidc_issuer=ISSUER, oidc_audience=AUDIENCE)
    app = create_app(settings, run_health_job=False)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://registry.test"
        ) as client:
            yield client


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@respx.mock
async def test_request_without_token_is_401(oidc_client: httpx.AsyncClient) -> None:
    response = await oidc_client.get("/api/v1/agents")
    assert response.status_code == 401


@respx.mock
async def test_owner_is_taken_from_sub(oidc_client: httpx.AsyncClient) -> None:
    _mock_issuer()
    token = make_token("alice", ["user"])
    response = await oidc_client.post(
        "/api/v1/agents", json=agent_entry_body(), headers=_auth(token)
    )
    assert response.status_code == 201
    assert response.json()["owner"] == "alice"


@respx.mock
async def test_users_see_only_their_own_entries(oidc_client: httpx.AsyncClient) -> None:
    _mock_issuer()
    alice, bob = make_token("alice", ["user"]), make_token("bob", ["user"])
    await oidc_client.post("/api/v1/agents", json=agent_entry_body("a1"), headers=_auth(alice))
    entry = (
        await oidc_client.post("/api/v1/agents", json=agent_entry_body("b1"), headers=_auth(bob))
    ).json()
    listing = (await oidc_client.get("/api/v1/agents", headers=_auth(alice))).json()
    assert [e["card"]["name"] for e in listing["items"]] == ["a1"]
    fetched = await oidc_client.get(f"/api/v1/agents/{entry['id']}", headers=_auth(alice))
    assert fetched.status_code == 404  # invisible, not forbidden


@respx.mock
async def test_admin_sees_all_and_can_delete(oidc_client: httpx.AsyncClient) -> None:
    _mock_issuer()
    alice = make_token("alice", ["user"])
    admin = make_token("root", ["admin", "user"])
    entry = (
        await oidc_client.post("/api/v1/agents", json=agent_entry_body("a1"), headers=_auth(alice))
    ).json()
    listing = (
        await oidc_client.get("/api/v1/agents", params={"owner": "all"}, headers=_auth(admin))
    ).json()
    assert listing["total"] == 1
    assert (
        await oidc_client.delete(f"/api/v1/agents/{entry['id']}", headers=_auth(admin))
    ).status_code == 204


@respx.mock
async def test_owner_all_requires_admin(oidc_client: httpx.AsyncClient) -> None:
    _mock_issuer()
    token = make_token("alice", ["user"])
    response = await oidc_client.get(
        "/api/v1/agents", params={"owner": "all"}, headers=_auth(token)
    )
    assert response.status_code == 403


@respx.mock
async def test_wrong_issuer_is_rejected(oidc_client: httpx.AsyncClient) -> None:
    _mock_issuer()
    token = make_token("alice", ["user"], issuer="http://evil.test/realms/x")
    response = await oidc_client.get("/api/v1/agents", headers=_auth(token))
    assert response.status_code == 401


@respx.mock
async def test_healthz_needs_no_auth(oidc_client: httpx.AsyncClient) -> None:
    assert (await oidc_client.get("/healthz")).status_code == 200


@respx.mock
async def test_admin_registrar_stamps_the_real_owner(oidc_client: httpx.AsyncClient) -> None:
    """A trusted admin service (the runtime) may publish on behalf of a user."""
    _mock_issuer()
    runtime_svc = make_token("runtime", ["admin"])
    body = {**agent_entry_body("a1"), "owner": "alice"}
    entry = (await oidc_client.post("/api/v1/agents", json=body, headers=_auth(runtime_svc))).json()
    assert entry["owner"] == "alice"
    # ...and alice sees it as hers
    alice = make_token("alice", ["user"])
    listing = (await oidc_client.get("/api/v1/agents", headers=_auth(alice))).json()
    assert [e["card"]["name"] for e in listing["items"]] == ["a1"]


@respx.mock
async def test_non_admin_cannot_spoof_owner(oidc_client: httpx.AsyncClient) -> None:
    _mock_issuer()
    mallory = make_token("mallory", ["user"])
    body = {**agent_entry_body("m1"), "owner": "alice"}
    entry = (await oidc_client.post("/api/v1/agents", json=body, headers=_auth(mallory))).json()
    assert entry["owner"] == "mallory"  # asserted owner ignored for non-admins


@respx.mock
async def test_teammates_see_and_manage_group_entries(oidc_client: httpx.AsyncClient) -> None:
    """Entries published into a team are visible and manageable for its members."""
    _mock_issuer()
    runtime_svc = make_token("runtime", ["admin"])
    body = {**agent_entry_body("pay-agent"), "owner": "user-x", "group": "team-payments"}
    entry = (await oidc_client.post("/api/v1/agents", json=body, headers=_auth(runtime_svc))).json()
    assert (entry["owner"], entry["group"]) == ("user-x", "team-payments")

    teammate = make_token("user-y", ["user"], groups=["team-payments"])
    stranger = make_token("user-z", ["user"], groups=["team-other"])

    listing = (await oidc_client.get("/api/v1/agents", headers=_auth(teammate))).json()
    assert [e["card"]["name"] for e in listing["items"]] == ["pay-agent"]
    search = (
        await oidc_client.get("/api/v1/agents/search", params={"q": "pay"}, headers=_auth(teammate))
    ).json()
    assert search["total"] == 1
    assert (
        await oidc_client.delete(f"/api/v1/agents/{entry['id']}", headers=_auth(teammate))
    ).status_code == 204

    # stranger never saw it (and it is gone now anyway)
    assert (await oidc_client.get("/api/v1/agents", headers=_auth(stranger))).json()["items"] == []
