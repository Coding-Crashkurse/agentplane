"""Add the ``enabled`` column (soft-disable) to ``entries``.

Introduced after 0.0.4: entries can be disabled instead of deleted. Disabled
entries are hidden from discovery (search) and skipped by the health job but
remain listed for their owner. The column is NOT NULL with a server default
of true so existing rows backfill to "enabled", matching the model's
Python-side default in ``db.py``.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "entries",
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_index(op.f("ix_entries_enabled"), "entries", ["enabled"])


def downgrade() -> None:
    op.drop_index(op.f("ix_entries_enabled"), table_name="entries")
    op.drop_column("entries", "enabled")
