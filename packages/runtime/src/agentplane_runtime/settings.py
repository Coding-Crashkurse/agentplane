"""Runtime configuration (SPEC §7.2), env prefix ``AGENTPLANE_RUNTIME_``."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

RUNTIME_VERSION = "0.0.9"

EPHEMERAL_TTL_S = 30 * 60  # SPEC §6.2: draft endpoints live 30 minutes


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTPLANE_RUNTIME_", extra="ignore", env_file=".env"
    )

    db_url: str = "sqlite+aiosqlite:///runtime.db"
    public_base_url: str = ""  # required to serve; validated on app start
    registry_url: str = ""  # required to self-register; empty disables registration
    registry_token: str = ""  # static bearer (expires); prefer client-credentials below
    # Service-account OIDC client-credentials for self-registration: auto-refreshed
    # and, unlike a static token, never expires out from under the runtime. The
    # service account needs the admin role in the registry to set owner/group
    # (SPEC §7.1). Issuer falls back to oidc_issuer when left empty.
    registry_client_id: str = ""
    registry_client_secret: str = ""
    registry_oidc_issuer: str = ""
    secret_key: str = ""  # Fernet key; required for resources with secrets
    # Pluggable secrets backend (SPEC_SAAS §11): only "fernet" ships here;
    # enterprise/KMS/Vault backends implement the same SecretsProvider and are
    # selected by name so call sites never change. Not a licensed feature —
    # at-rest security is not paywalled.
    secrets_backend: str = "fernet"
    llm_base_url: str = ""  # gateway's OpenAI-compatible endpoint (resource default)
    auth_mode: Literal["none", "oidc"] = "none"
    oidc_issuer: str = ""
    oidc_audience: str = ""
    roles_claim: str = "realm_access.roles"
    username_claim: str = "preferred_username"  # display name recorded on definitions/entries
    groups_claim: str = "groups"  # team membership; scopes shared resources
    admin_role: str = "admin"
    builder_role: str = "builder"  # role required for definition/resource writes (SPEC §7.1)
    http_timeout_s: float = 60.0
    # Recursion guard for orchestrators: an agent may call an agent (over A2A),
    # which may call another. The depth travels in the A2A message metadata;
    # requests deeper than this fail instead of looping A -> B -> A forever.
    max_agent_call_depth: int = 5
    host: str = "0.0.0.0"
    port: int = 8000
    # A2A task persistence (SPEC §6.5): "database" stores tasks in db_url
    # (SQLite standalone, Postgres in compose) so conversations survive
    # restarts and clients can restore history via ListTasks/GetTask.
    # "memory" keeps today's per-process behavior. Draft endpoints always
    # stay in-memory.
    task_store: Literal["memory", "database"] = "memory"
    # SaaS guard-rail: cap how many non-ephemeral definitions one owner may keep
    # deployed at once. 0 = unlimited (default). Admins bypass; redeploying an
    # already-deployed definition does not consume an extra slot (SPEC §7.2).
    max_deployments_per_owner: int = 0
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
