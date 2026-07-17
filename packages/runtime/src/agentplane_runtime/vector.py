"""Read-only vector DB access (SPEC §3.2): qdrant via REST, pgvector via asyncpg.

Vector DBs are consumed **read-only by contract** — there is no upsert path
anywhere in the runtime.
"""

from __future__ import annotations

from importlib.util import find_spec

import httpx

from agentplane_core import Document, JsonObject, VectorDBResource


class VectorDBError(RuntimeError):
    """Vector DB request failed."""


DEFAULT_TIMEOUT_S = 30.0


class QdrantReader:
    """Qdrant REST access: collection info + similarity search."""

    def __init__(self, url: str, api_key: str = "", *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._url = url.rstrip("/")
        self._headers = {"api-key": api_key} if api_key else {}
        self._timeout = timeout

    async def collection_dimension(self, collection: str) -> int:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{self._url}/collections/{collection}", headers=self._headers
                )
        except httpx.HTTPError as exc:
            raise VectorDBError(f"qdrant unreachable: {exc}") from exc
        if response.status_code != httpx.codes.OK:
            raise VectorDBError(
                f"cannot read collection {collection!r}: HTTP {response.status_code}"
            )
        params = response.json().get("result", {}).get("config", {}).get("params", {})
        vectors = params.get("vectors", {})
        size = vectors.get("size") if isinstance(vectors, dict) else None
        if not isinstance(size, int):
            raise VectorDBError(f"collection {collection!r} has no readable vector size")
        return size

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        filter: JsonObject | None = None,
        *,
        min_score: float | None = None,
    ) -> list[Document]:
        body: dict[str, object] = {"vector": vector, "limit": top_k, "with_payload": True}
        if filter is not None:
            body["filter"] = filter
        if min_score is not None:
            body["score_threshold"] = min_score
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._url}/collections/{collection}/points/search",
                    json=body,
                    headers=self._headers,
                )
        except httpx.HTTPError as exc:
            raise VectorDBError(f"qdrant unreachable: {exc}") from exc
        if response.status_code != httpx.codes.OK:
            raise VectorDBError(f"search failed: HTTP {response.status_code}")
        documents: list[Document] = []
        for hit in response.json().get("result", []):
            payload = hit.get("payload") or {}
            text = payload.get("text") or payload.get("content") or payload.get("chunk") or ""
            metadata = {k: v for k, v in payload.items() if k not in ("text", "content", "chunk")}
            documents.append(
                Document(text=str(text), score=float(hit.get("score", 0.0)), metadata=metadata)
            )
        return documents


class PgvectorReader:
    """pgvector access via asyncpg ([postgres] extra); degrades with a clear error."""

    def __init__(self, dsn: str, *, table_prefix: str = "", timeout: float | None = None) -> None:
        if find_spec("asyncpg") is None:
            raise VectorDBError("pgvector resources need the [postgres] extra (asyncpg) installed")
        self._dsn = dsn
        self._table_prefix = table_prefix
        self._timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_S

    async def _connect(self) -> object:
        import asyncpg  # noqa: PLC0415 - optional [postgres] extra

        try:
            return await asyncpg.connect(self._dsn, timeout=self._timeout)
        except (OSError, asyncpg.PostgresError) as exc:
            raise VectorDBError(f"pgvector unreachable: {exc}") from exc

    async def collection_dimension(self, collection: str) -> int:
        import asyncpg  # noqa: PLC0415 - optional [postgres] extra

        conn = await self._connect()
        try:
            row = await conn.fetchrow(  # type: ignore[attr-defined]
                """
                SELECT atttypmod AS dim FROM pg_attribute
                WHERE attrelid = $1::regclass AND attname = 'embedding'
                """,
                collection,
            )
        except asyncpg.PostgresError as exc:
            raise VectorDBError(f"cannot read collection {collection!r}: {exc}") from exc
        finally:
            await conn.close()  # type: ignore[attr-defined]
        if row is None or not isinstance(row["dim"], int) or row["dim"] <= 0:
            raise VectorDBError(f"table {collection!r} has no readable embedding dimension")
        return int(row["dim"])

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        filter: JsonObject | None = None,
        *,
        min_score: float | None = None,
    ) -> list[Document]:
        if not collection.replace("_", "").isalnum():
            raise VectorDBError(f"invalid collection name {collection!r}")
        vector_literal = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
        conn = await self._connect()
        try:
            rows = await conn.fetch(  # type: ignore[attr-defined]
                f"""
                SELECT text, metadata, 1 - (embedding <=> $1::vector) AS score
                FROM {collection} ORDER BY embedding <=> $1::vector LIMIT $2
                """,
                vector_literal,
                top_k,
            )
        finally:
            await conn.close()  # type: ignore[attr-defined]
        documents: list[Document] = []
        for row in rows:
            score = float(row["score"])
            if min_score is not None and score < min_score:
                continue
            metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
            documents.append(Document(text=str(row["text"]), score=score, metadata=metadata))
        return documents


Reader = QdrantReader | PgvectorReader


async def reader_for(
    resource: VectorDBResource,
    *,
    api_key: str = "",
    dsn: str = "",
    timeout: float | None = None,
) -> Reader:
    """Build the right reader for a VectorDB resource.

    ``timeout`` bounds the connection/request: definition validation passes a
    short timeout so a slow or unreachable DB fails fast (surfacing as a
    ``VectorDBError`` the E022 check treats as "no answer", not an error);
    execution leaves it at the reader default.
    """
    reader_timeout = DEFAULT_TIMEOUT_S if timeout is None else timeout
    if resource.kind == "qdrant":
        return QdrantReader(resource.url, api_key, timeout=reader_timeout)
    return PgvectorReader(dsn or resource.url, timeout=reader_timeout)


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "PgvectorReader",
    "QdrantReader",
    "Reader",
    "VectorDBError",
    "reader_for",
]
