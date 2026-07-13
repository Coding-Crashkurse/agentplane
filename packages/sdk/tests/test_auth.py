"""Token providers: static + OIDC client credentials with caching."""

from __future__ import annotations

import respx
from httpx import Response

from agentplane_sdk import OidcClientCredentialsProvider, StaticTokenProvider, as_token_provider

ISSUER = "http://keycloak.test/realms/agentplane"


async def test_static_token_provider() -> None:
    provider = StaticTokenProvider("abc")
    assert await provider.get_token() == "abc"


def test_as_token_provider_coercion() -> None:
    assert as_token_provider(None) is None
    provider = as_token_provider("tok")
    assert isinstance(provider, StaticTokenProvider)
    passthrough = as_token_provider(provider)
    assert passthrough is provider


@respx.mock
async def test_oidc_client_credentials_caches_token() -> None:
    respx.get(f"{ISSUER}/.well-known/openid-configuration").mock(
        return_value=Response(200, json={"token_endpoint": f"{ISSUER}/token"})
    )
    token_route = respx.post(f"{ISSUER}/token").mock(
        return_value=Response(200, json={"access_token": "jwt-1", "expires_in": 300})
    )
    provider = OidcClientCredentialsProvider(ISSUER, "cli", "s3cret")
    assert await provider.get_token() == "jwt-1"
    assert await provider.get_token() == "jwt-1"  # cached
    assert token_route.call_count == 1
    body = token_route.calls.last.request.content.decode()
    assert "grant_type=client_credentials" in body
    assert "client_id=cli" in body


@respx.mock
async def test_oidc_token_refreshes_after_expiry() -> None:
    respx.get(f"{ISSUER}/.well-known/openid-configuration").mock(
        return_value=Response(200, json={"token_endpoint": f"{ISSUER}/token"})
    )
    token_route = respx.post(f"{ISSUER}/token").mock(
        return_value=Response(200, json={"access_token": "jwt-x", "expires_in": 1})
    )
    provider = OidcClientCredentialsProvider(ISSUER, "cli", "s3cret")
    await provider.get_token()
    provider._expires_at = 0.0  # simulate expiry
    await provider.get_token()
    assert token_route.call_count == 2
