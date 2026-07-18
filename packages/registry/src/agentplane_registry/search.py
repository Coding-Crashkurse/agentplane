"""Search backend (SPEC §5.1): text match + optional semantic with RRF merge.

Text search: case-insensitive match over name, description and skill
names/descriptions plus AND tag filter. Semantic search: cosine top-k over
the entry embedding, merged with the text hits via reciprocal rank fusion.
Without the ``[semantic]`` extra (numpy) or embeddings config, semantic
requests silently degrade to text search (the API adds ``X-Degraded``).
"""

from __future__ import annotations

import logging
import math
from importlib.util import find_spec
from uuid import UUID

from sqlalchemy import delete, or_, select

from agentplane_core import (
    Page,
    RegistryEntry,
    SearchBackend,
    SearchQuery,
    ToolCard,
)
from agentplane_registry.db import Database, EntryEmbeddingRow, EntryRow, row_to_entry
from agentplane_registry.embeddings import EmbeddingsClient, embedding_text

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard reciprocal-rank-fusion constant


def semantic_available() -> bool:
    """The [semantic] extra is installed when numpy imports."""
    return find_spec("numpy") is not None


def _haystack(entry: RegistryEntry) -> str:
    card = entry.card
    parts = [card.name, card.description]
    if isinstance(card, ToolCard):
        parts += [f"{tool.name} {tool.description}" for tool in card.tools]
    else:
        parts += [f"{skill.name} {skill.description}" for skill in card.skills]
    return "\n".join(parts).lower()


def _matches(entry: RegistryEntry, query: SearchQuery) -> bool:
    if query.tags and not set(query.tags) <= set(entry.tags):
        return False
    return not query.q or query.q.lower() in _haystack(entry)


class RegistrySearch(SearchBackend):
    """SQL-filtered text search with optional in-process semantic ranking."""

    def __init__(self, db: Database, embeddings: EmbeddingsClient | None) -> None:
        self._db = db
        self._embeddings = embeddings

    @property
    def semantic_enabled(self) -> bool:
        return self._embeddings is not None and semantic_available()

    async def index(self, entry: RegistryEntry) -> None:
        """(Re-)embed an entry; no-op without semantic support."""
        if self._embeddings is None or not semantic_available():
            return
        try:
            vector = await self._embeddings.embed(embedding_text(entry))
        except Exception:
            logger.warning("embedding failed for entry %s", entry.id, exc_info=True)
            return
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        async with self._db.session() as session, session.begin():
            existing = await session.get(EntryEmbeddingRow, str(entry.id))
            if existing is None:
                session.add(EntryEmbeddingRow(entry_id=str(entry.id), vector=vector, norm=norm))
            else:
                existing.vector = vector
                existing.norm = norm

    async def remove(self, entry_id: UUID) -> None:
        async with self._db.session() as session, session.begin():
            await session.execute(
                delete(EntryEmbeddingRow).where(EntryEmbeddingRow.entry_id == str(entry_id))
            )

    async def _load_candidates(self, query: SearchQuery) -> list[RegistryEntry]:
        stmt = select(EntryRow).order_by(EntryRow.created_at.desc())
        if not query.include_disabled:
            stmt = stmt.where(EntryRow.enabled.is_(True))
        if query.kind is not None:
            stmt = stmt.where(EntryRow.kind == query.kind)
        if query.status is not None:
            stmt = stmt.where(EntryRow.status == query.status)
        if query.owner is not None:
            conditions = [EntryRow.owner == query.owner]
            if query.groups:
                conditions.append(EntryRow.group.in_(query.groups))
            stmt = stmt.where(or_(*conditions))
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [row_to_entry(row) for row in rows]

    async def _semantic_ranking(self, query: SearchQuery) -> list[UUID]:
        """Entry ids ranked by cosine similarity to the query embedding."""
        if self._embeddings is None:
            return []
        query_vector = await self._embeddings.embed(query.q)
        query_norm = math.sqrt(sum(v * v for v in query_vector)) or 1.0
        async with self._db.session() as session:
            rows = (await session.execute(select(EntryEmbeddingRow))).scalars().all()
        if not rows:
            return []
        import numpy as np  # noqa: PLC0415 - [semantic] extra, imported only when present

        matrix = np.array([row.vector for row in rows], dtype=np.float64)
        norms = np.array([row.norm for row in rows], dtype=np.float64)
        scores = (matrix @ np.array(query_vector)) / (norms * query_norm)
        order = np.argsort(-scores)
        return [UUID(rows[int(i)].entry_id) for i in order]

    async def search(self, q: SearchQuery) -> Page:
        candidates = await self._load_candidates(q)
        text_hits = [entry for entry in candidates if _matches(entry, q)]

        use_semantic = q.semantic and self.semantic_enabled and bool(q.q)
        if use_semantic:
            try:
                semantic_ids = await self._semantic_ranking(q)
            except Exception:
                logger.warning("semantic search failed; degrading to text", exc_info=True)
                semantic_ids = []
            by_id = {entry.id: entry for entry in candidates if _matches_filters(entry, q)}
            ranked = _rrf_merge(
                [entry.id for entry in text_hits],
                [entry_id for entry_id in semantic_ids if entry_id in by_id],
            )
            hits = [by_id[entry_id] for entry_id in ranked if entry_id in by_id]
        else:
            hits = text_hits

        total = len(hits)
        page_items = hits[q.offset : q.offset + q.limit]
        return Page(items=page_items, total=total, limit=q.limit, offset=q.offset)


def _matches_filters(entry: RegistryEntry, query: SearchQuery) -> bool:
    """Tag filter only (semantic hits skip the text match but not the filters)."""
    return not query.tags or set(query.tags) <= set(entry.tags)


def _rrf_merge(first: list[UUID], second: list[UUID]) -> list[UUID]:
    """Reciprocal rank fusion of two rankings."""
    scores: dict[UUID, float] = {}
    for ranking in (first, second):
        for rank, entry_id in enumerate(ranking):
            scores[entry_id] = scores.get(entry_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda entry_id: -scores[entry_id])


__all__ = ["RegistrySearch", "semantic_available"]
