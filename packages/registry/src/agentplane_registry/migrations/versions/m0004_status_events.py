"""Add the ``entry_status_events`` table (status history, SPEC §5.3).

Introduced after 0.0.4: entries record their status transitions so the API
can answer "was this agent up in the last 24h?". Rows past the retention
window are pruned by the health job.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entry_status_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("entry_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f("ix_entry_status_events_entry_id"), "entry_status_events", ["entry_id"])
    op.create_index(op.f("ix_entry_status_events_at"), "entry_status_events", ["at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_entry_status_events_at"), table_name="entry_status_events")
    op.drop_index(op.f("ix_entry_status_events_entry_id"), table_name="entry_status_events")
    op.drop_table("entry_status_events")
