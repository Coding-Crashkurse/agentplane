"""Secrets backend selection (SPEC_SAAS §11: pluggable KMS/Vault seam)."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from agentplane_runtime.db import Database
from agentplane_runtime.secrets import FernetSecretsProvider, make_secrets_provider
from agentplane_runtime.settings import RuntimeSettings


def test_fernet_backend_returns_fernet_provider() -> None:
    key = Fernet.generate_key().decode()
    settings = RuntimeSettings(public_base_url="http://localhost", secret_key=key)
    provider = make_secrets_provider(settings, Database("sqlite+aiosqlite:///:memory:"))
    assert isinstance(provider, FernetSecretsProvider)


def test_unknown_backend_raises() -> None:
    settings = RuntimeSettings(
        public_base_url="http://localhost", secret_key="x", secrets_backend="vault"
    )
    with pytest.raises(RuntimeError, match="unknown secrets backend"):
        make_secrets_provider(settings, Database("sqlite+aiosqlite:///:memory:"))


def test_fernet_backend_requires_key() -> None:
    settings = RuntimeSettings(public_base_url="http://localhost", secret_key="")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        make_secrets_provider(settings, Database("sqlite+aiosqlite:///:memory:"))
