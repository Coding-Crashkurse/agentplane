"""Add the ``group`` column (team sharing) to ``definitions`` and ``resources``.

Introduced after 0.0.2: definitions and resources gained a group scope for
team sharing. The columns are NOT NULL with a server default of ``''`` so
existing rows backfill to "no group", matching the models' Python-side
defaults in ``db.py``.

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
    op.add_column(
        "definitions", sa.Column("group", sa.String(255), nullable=False, server_default="")
    )
    op.create_index(op.f("ix_definitions_group"), "definitions", ["group"])
    op.add_column(
        "resources", sa.Column("group", sa.String(255), nullable=False, server_default="")
    )
    op.create_index(op.f("ix_resources_group"), "resources", ["group"])


def downgrade() -> None:
    op.drop_index(op.f("ix_resources_group"), table_name="resources")
    op.drop_column("resources", "group")
    op.drop_index(op.f("ix_definitions_group"), table_name="definitions")
    op.drop_column("definitions", "group")
