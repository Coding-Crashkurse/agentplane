"""Baseline: the registry schema exactly as released in agentplane-registry 0.0.2.

Deliberately NOT the current ``db.py`` metadata: 0.0.2 predates the ``group``
column (added by revision 0002), and a real 0.0.2 database gets stamped at this
revision, so the baseline must match what 0.0.2's ``create_all`` produced.

``entry_embeddings`` is part of the baseline on every backend: SPEC §5.2 scopes
it to [semantic]+Postgres, but ``db.py`` deliberately persists JSON vectors on
all backends (so SQLite restarts do not re-embed) and searches with the
in-process numpy fallback. No pgvector DDL lives in this baseline — a
pgvector-backed column would arrive as a later migration guarded by dialect
and the installed extra.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("card_json", sa.String(), nullable=False),
        sa.Column("tags_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner", "name", name="uq_entries_owner_name"),
    )
    op.create_index(op.f("ix_entries_kind"), "entries", ["kind"])
    op.create_index(op.f("ix_entries_name"), "entries", ["name"])
    op.create_index(op.f("ix_entries_owner"), "entries", ["owner"])
    op.create_index(op.f("ix_entries_status"), "entries", ["status"])
    op.create_table(
        "entry_embeddings",
        sa.Column("entry_id", sa.String(36), primary_key=True),
        sa.Column("vector", sa.JSON(), nullable=False),
        sa.Column("norm", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("entry_embeddings")
    op.drop_index(op.f("ix_entries_status"), table_name="entries")
    op.drop_index(op.f("ix_entries_owner"), table_name="entries")
    op.drop_index(op.f("ix_entries_name"), table_name="entries")
    op.drop_index(op.f("ix_entries_kind"), table_name="entries")
    op.drop_table("entries")
