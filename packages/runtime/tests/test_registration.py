"""Registry self-registration auth selection (SPEC §6.5, §7.1).

Regression for the runtime sending no bearer to the registry (HTTP 401), which
left every deployment unregistered (registry_id null).
"""

from __future__ import annotations

from agentplane_runtime.registration import RegistryRegistrar
from agentplane_runtime.settings import RuntimeSettings
from agentplane_sdk import OidcClientCredentialsProvider


def _settings(**kw: object) -> RuntimeSettings:
    return RuntimeSettings(_env_file=None, registry_url="http://registry:8100", **kw)  # type: ignore[arg-type]


def test_prefers_client_credentials_over_static_token() -> None:
    registrar = RegistryRegistrar(
        _settings(
            registry_token="ignored-static",
            registry_client_id="svc",
            registry_client_secret="sekret",
            oidc_issuer="http://auth/realms/agentplane",
        )
    )
    assert isinstance(registrar._auth, OidcClientCredentialsProvider)


def test_client_credentials_issuer_falls_back_to_oidc_issuer() -> None:
    registrar = RegistryRegistrar(
        _settings(
            registry_client_id="svc",
            registry_client_secret="sekret",
            oidc_issuer="http://auth/realms/agentplane",
        )
    )
    assert isinstance(registrar._auth, OidcClientCredentialsProvider)
    # dedicated issuer wins when set
    other = RegistryRegistrar(
        _settings(
            registry_client_id="svc",
            registry_client_secret="sekret",
            registry_oidc_issuer="http://other/realms/x",
            oidc_issuer="http://auth/realms/agentplane",
        )
    )
    assert isinstance(other._auth, OidcClientCredentialsProvider)


def test_static_token_when_no_client_credentials() -> None:
    registrar = RegistryRegistrar(_settings(registry_token="static-abc"))
    assert registrar._auth == "static-abc"


def test_no_auth_when_unconfigured() -> None:
    assert RegistryRegistrar(_settings())._auth is None
