"""Token providers: static bearer and OIDC client-credentials (SPEC §4.1)."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import httpx

from agentplane_sdk.errors import AuthError, TransportError


@runtime_checkable
class TokenProvider(Protocol):
    """Anything that can produce a bearer token."""

    async def get_token(self) -> str: ...


class StaticTokenProvider:
    """Wraps a fixed bearer token."""

    def __init__(self, token: str) -> None:
        self._token = token

    async def get_token(self) -> str:
        return self._token


class OidcClientCredentialsProvider:
    """Fetches tokens via the OIDC client-credentials grant, cached until expiry."""

    def __init__(
        self,
        issuer: str,
        client_id: str,
        client_secret: str,
        *,
        audience: str | None = None,
        leeway_s: float = 30.0,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._client_id = client_id
        self._client_secret = client_secret
        self._audience = audience
        self._leeway_s = leeway_s
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._token_endpoint: str | None = None

    async def get_token(self) -> str:
        if self._token is not None and time.monotonic() < self._expires_at:
            return self._token
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if self._token_endpoint is None:
                    discovery = await client.get(f"{self._issuer}/.well-known/openid-configuration")
                    discovery.raise_for_status()
                    endpoint = discovery.json().get("token_endpoint")
                    if not isinstance(endpoint, str):
                        raise AuthError("issuer discovery returned no token_endpoint")
                    self._token_endpoint = endpoint
                data = {
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                }
                if self._audience:
                    data["audience"] = self._audience
                response = await client.post(self._token_endpoint, data=data)
        except httpx.HTTPError as exc:
            raise TransportError(f"token request failed: {exc}") from exc
        if response.status_code != httpx.codes.OK:
            raise AuthError(f"token endpoint returned {response.status_code}")
        payload = response.json()
        token = payload.get("access_token")
        if not isinstance(token, str):
            raise AuthError("token endpoint returned no access_token")
        expires_in = payload.get("expires_in", 300)
        ttl = float(expires_in) if isinstance(expires_in, int | float) else 300.0
        self._token = token
        self._expires_at = time.monotonic() + max(ttl - self._leeway_s, 5.0)
        return token


def as_token_provider(token: str | TokenProvider | None) -> TokenProvider | None:
    if token is None:
        return None
    if isinstance(token, str):
        return StaticTokenProvider(token)
    return token


__all__ = [
    "OidcClientCredentialsProvider",
    "StaticTokenProvider",
    "TokenProvider",
    "as_token_provider",
]
