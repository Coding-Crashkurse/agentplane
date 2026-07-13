"""CLI configuration resolution (SPEC §4.2).

Order: flags → env (``AGENTPLANE_*``) → ``~/.config/agentplane/config.toml``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentplane_sdk.auth import OidcClientCredentialsProvider, TokenProvider, as_token_provider

CONFIG_PATH = Path.home() / ".config" / "agentplane" / "config.toml"


class CliConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    runtime_url: str | None = None
    registry_url: str | None = None
    token: str | None = None
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    oidc_audience: str | None = None


def _load_config_file(path: Path) -> CliConfig:
    if not path.is_file():
        return CliConfig()
    with path.open("rb") as fh:
        return CliConfig.model_validate(tomllib.load(fh))


def resolve_config(
    *,
    runtime_url: str | None = None,
    registry_url: str | None = None,
    token: str | None = None,
    config_path: Path | None = None,
) -> CliConfig:
    """Merge flags, environment and config file into one effective config."""
    file_config = _load_config_file(config_path or CONFIG_PATH)

    def pick(flag: str | None, env_name: str, file_value: str | None) -> str | None:
        return flag or os.environ.get(env_name) or file_value

    return CliConfig(
        runtime_url=pick(runtime_url, "AGENTPLANE_RUNTIME_URL", file_config.runtime_url),
        registry_url=pick(registry_url, "AGENTPLANE_REGISTRY_URL", file_config.registry_url),
        token=pick(token, "AGENTPLANE_TOKEN", file_config.token),
        oidc_issuer=pick(None, "AGENTPLANE_OIDC_ISSUER", file_config.oidc_issuer),
        oidc_client_id=pick(None, "AGENTPLANE_OIDC_CLIENT_ID", file_config.oidc_client_id),
        oidc_client_secret=pick(
            None, "AGENTPLANE_OIDC_CLIENT_SECRET", file_config.oidc_client_secret
        ),
        oidc_audience=pick(None, "AGENTPLANE_OIDC_AUDIENCE", file_config.oidc_audience),
    )


def token_provider_from_config(config: CliConfig) -> TokenProvider | None:
    """Static token wins; otherwise OIDC client credentials when configured."""
    if config.token:
        return as_token_provider(config.token)
    if config.oidc_issuer and config.oidc_client_id and config.oidc_client_secret:
        return OidcClientCredentialsProvider(
            config.oidc_issuer,
            config.oidc_client_id,
            config.oidc_client_secret,
            audience=config.oidc_audience,
        )
    return None


__all__ = ["CONFIG_PATH", "CliConfig", "resolve_config", "token_provider_from_config"]
