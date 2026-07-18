"""Default SecretsProvider: Fernet-encrypted DB column (SPEC §3.3)."""

from __future__ import annotations

from cryptography.fernet import Fernet

from agentplane_core import SecretsProvider
from agentplane_runtime.db import Database, SecretRow
from agentplane_runtime.settings import RuntimeSettings


class SecretNotFoundError(KeyError):
    """No secret stored under the given ref."""


class FernetSecretsProvider(SecretsProvider):
    """Encrypts values with ``AGENTPLANE_RUNTIME_SECRET_KEY`` at rest."""

    def __init__(self, db: Database, secret_key: str) -> None:
        self._db = db
        self._fernet = Fernet(secret_key.encode("ascii"))

    async def put(self, ref: str, value: str) -> None:
        ciphertext = self._fernet.encrypt(value.encode("utf-8")).decode("ascii")
        async with self._db.session() as session, session.begin():
            row = await session.get(SecretRow, ref)
            if row is None:
                session.add(SecretRow(ref=ref, ciphertext=ciphertext))
            else:
                row.ciphertext = ciphertext

    async def get(self, ref: str) -> str:
        async with self._db.session() as session:
            row = await session.get(SecretRow, ref)
        if row is None:
            raise SecretNotFoundError(ref)
        return self._fernet.decrypt(row.ciphertext.encode("ascii")).decode("utf-8")

    async def delete(self, ref: str) -> None:
        async with self._db.session() as session, session.begin():
            row = await session.get(SecretRow, ref)
            if row is not None:
                await session.delete(row)


def make_secrets_provider(settings: RuntimeSettings, db: Database) -> SecretsProvider:
    """Select the secrets backend by name (SPEC_SAAS §11: pluggable KMS/Vault seam).

    Only ``fernet`` ships in the community package. Enterprise/KMS/Vault backends
    are optional extras that implement the same :class:`SecretsProvider` and plug
    in here — the call sites (resource CRUD) never change. This is the extension
    point, not a licensed feature: at-rest encryption is never paywalled.
    """
    backend = settings.secrets_backend
    if backend == "fernet":
        if not settings.secret_key:
            raise RuntimeError("AGENTPLANE_RUNTIME_SECRET_KEY is required for the fernet backend")
        return FernetSecretsProvider(db, settings.secret_key)
    raise RuntimeError(f"unknown secrets backend {backend!r}")


__all__ = ["FernetSecretsProvider", "SecretNotFoundError", "make_secrets_provider"]
