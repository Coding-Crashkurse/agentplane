"""Registry configuration (SPEC §7.2), env prefix ``AGENTPLANE_REGISTRY_``."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REGISTRY_VERSION = "0.0.5"


class RegistrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTPLANE_REGISTRY_", extra="ignore")

    db_url: str = "sqlite+aiosqlite:///registry.db"
    auth_mode: Literal["none", "oidc"] = "none"
    oidc_issuer: str = ""
    oidc_audience: str = ""
    roles_claim: str = "realm_access.roles"
    groups_claim: str = "groups"  # team membership; scopes shared entries
    admin_role: str = "admin"
    health_interval_s: float = 60.0
    health_mcp: bool = True
    health_timeout_s: float = 10.0
    history_retention_h: float = 168.0  # 7 days of status transitions
    embeddings_base_url: str = ""
    embeddings_model: str = ""
    allow_private_urls: bool = False


__all__ = ["REGISTRY_VERSION", "RegistrySettings"]
