"""Runtime configuration (SPEC §7.2), env prefix ``AGENTPLANE_RUNTIME_``."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RUNTIME_VERSION = "0.0.1"

EPHEMERAL_TTL_S = 30 * 60  # SPEC §6.2: draft endpoints live 30 minutes


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTPLANE_RUNTIME_", extra="ignore")

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
    admin_role: str = "admin"
    http_timeout_s: float = 60.0


__all__ = ["EPHEMERAL_TTL_S", "RUNTIME_VERSION", "RuntimeSettings"]
