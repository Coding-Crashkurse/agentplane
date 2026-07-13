"""FernetSecretsProvider round-trip and ciphertext-at-rest."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from agentplane_runtime.db import Database, SecretRow
from agentplane_runtime.secrets import FernetSecretsProvider, SecretNotFoundError


async def test_put_get_delete_roundtrip() -> None:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    provider = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    await provider.put("resource/default-llm/api_key_secret", "sk-value-1")
    assert await provider.get("resource/default-llm/api_key_secret") == "sk-value-1"
    await provider.put("resource/default-llm/api_key_secret", "sk-value-2")  # overwrite
    assert await provider.get("resource/default-llm/api_key_secret") == "sk-value-2"
    await provider.delete("resource/default-llm/api_key_secret")
    with pytest.raises(SecretNotFoundError):
        await provider.get("resource/default-llm/api_key_secret")


async def test_value_is_encrypted_at_rest() -> None:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    provider = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    await provider.put("ref", "plaintext-secret")
    async with db.session() as session:
        row = (await session.execute(select(SecretRow))).scalar_one()
    assert "plaintext-secret" not in row.ciphertext
