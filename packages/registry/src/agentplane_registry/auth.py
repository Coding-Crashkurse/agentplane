"""Optional OIDC auth (SPEC §5.5): any issuer, JWKS cache, role mapping."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx
import jwt
from fastapi import HTTPException, Request, status
from jwt.types import Options

from agentplane_registry.settings import RegistrySettings


@dataclass(frozen=True)
class Principal:
    """Authenticated caller."""

    sub: str
    roles: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False


ANONYMOUS = Principal(sub="anonymous", roles=frozenset(), is_admin=False)


def _claim_path(claims: dict[str, object], path: str) -> object:
    value: object = claims
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


class OidcValidator:
    """Validates JWTs against a generic OIDC issuer (discovery + JWKS cache)."""

    def __init__(
        self,
        issuer: str,
        audience: str,
        roles_claim: str,
        admin_role: str,
        *,
        jwks_ttl_s: float = 300.0,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._roles_claim = roles_claim
        self._admin_role = admin_role
        self._jwks_ttl_s = jwks_ttl_s
        self._jwks: dict[str, jwt.PyJWK] = {}
        self._jwks_fetched_at = 0.0

    async def _refresh_jwks(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            discovery = await client.get(f"{self._issuer}/.well-known/openid-configuration")
            discovery.raise_for_status()
            jwks_uri = discovery.json().get("jwks_uri")
            if not isinstance(jwks_uri, str):
                raise RuntimeError("issuer discovery returned no jwks_uri")
            jwks_response = await client.get(jwks_uri)
            jwks_response.raise_for_status()
        keys = jwt.PyJWKSet.from_dict(jwks_response.json()).keys
        self._jwks = {key.key_id: key for key in keys if key.key_id}
        self._jwks_fetched_at = time.monotonic()

    async def _key_for(self, kid: str) -> jwt.PyJWK:
        stale = time.monotonic() - self._jwks_fetched_at > self._jwks_ttl_s
        if kid not in self._jwks or stale:
            await self._refresh_jwks()
        key = self._jwks.get(kid)
        if key is None:
            raise jwt.InvalidTokenError(f"unknown key id {kid!r}")
        return key

    async def validate(self, token: str) -> Principal:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not isinstance(kid, str):
            raise jwt.InvalidTokenError("token has no kid header")
        key = await self._key_for(kid)
        options: Options = {"verify_aud": bool(self._audience)}
        claims = jwt.decode(
            token,
            key=key.key,
            algorithms=["RS256", "ES256"],
            audience=self._audience or None,
            issuer=self._issuer,
            options=options,
        )
        roles_value = _claim_path(claims, self._roles_claim)
        roles = (
            frozenset(str(role) for role in roles_value)
            if isinstance(roles_value, list)
            else frozenset()
        )
        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub:
            raise jwt.InvalidTokenError("token has no sub")
        return Principal(sub=sub, roles=roles, is_admin=self._admin_role in roles)


class Authenticator:
    """Resolves the request principal according to AUTH_MODE."""

    def __init__(self, settings: RegistrySettings) -> None:
        self.mode = settings.auth_mode
        self._validator: OidcValidator | None = None
        if self.mode == "oidc":
            if not settings.oidc_issuer:
                raise RuntimeError("AUTH_MODE=oidc requires AGENTPLANE_REGISTRY_OIDC_ISSUER")
            self._validator = OidcValidator(
                settings.oidc_issuer,
                settings.oidc_audience,
                settings.roles_claim,
                settings.admin_role,
            )

    async def authenticate(self, request: Request) -> Principal:
        if self.mode == "none" or self._validator is None:
            return ANONYMOUS
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
        try:
            return await self._validator.validate(token)
        except jwt.InvalidTokenError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"issuer unreachable: {exc}"
            ) from exc


__all__ = ["ANONYMOUS", "Authenticator", "OidcValidator", "Principal"]
