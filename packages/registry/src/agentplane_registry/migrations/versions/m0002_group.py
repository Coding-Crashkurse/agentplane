"""Add the ``group`` column (team sharing) to ``entries``.

Introduced after 0.0.2: entries gained a group scope for team sharing. The
column is NOT NULL with a server default of ``''`` so existing rows backfill
to "no group", matching the model's Python-side default in ``db.py``.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("entries", sa.Column("group", sa.String(255), nullable=False, server_default=""))
    op.create_index(op.f("ix_entries_group"), "entries", ["group"])


def downgrade() -> None:
    op.drop_index(op.f("ix_entries_group"), table_name="entries")
    op.drop_column("entries", "group")
