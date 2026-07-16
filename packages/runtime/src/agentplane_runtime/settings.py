"""Runtime configuration (SPEC §7.2), env prefix ``AGENTPLANE_RUNTIME_``."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

RUNTIME_VERSION = "0.0.3"

EPHEMERAL_TTL_S = 30 * 60  # SPEC §6.2: draft endpoints live 30 minutes


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTPLANE_RUNTIME_", extra="ignore", env_file=".env"
    )

    db_url: str = "sqlite+aiosqlite:///runtime.db"
    public_base_url: str = ""  # required to serve; validated on app start
    registry_url: str = ""  # required to self-register; empty disables registration
    registry_token: str = ""
    secret_key: str = ""  # Fernet key; required for resources with secrets
    llm_base_url: str = ""  # gateway's OpenAI-compatible endpoint (resource default)
    auth_mode: Literal["none", "oidc"] = "none"
    oidc_issuer: str = ""
    oidc_audience: str = ""
    roles_claim: str = "realm_access.roles"
    groups_claim: str = "groups"  # team membership; scopes shared resources
    admin_role: str = "admin"
    http_timeout_s: float = 60.0
    host: str = "0.0.0.0"
    port: int = 8000
    # Browsers only reach the endpoints directly when the runtime runs without a
    # gateway (local builder playground). In production agentgateway owns CORS,
    # so this stays empty and no middleware is installed. "*" allows any origin.
    cors_origins: Annotated[list[str], NoDecode] = []

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated env value (``a.example,b.example``) or a JSON list."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


__all__ = ["EPHEMERAL_TTL_S", "RUNTIME_VERSION", "RuntimeSettings"]
