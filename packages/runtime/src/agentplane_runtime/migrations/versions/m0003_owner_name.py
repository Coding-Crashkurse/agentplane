"""Add the ``owner_name`` display column to ``definitions``.

Introduced after 0.0.5-dev: ``owner`` (the OIDC subject) stays the
authorization key; ``owner_name`` records the creator's display name
(e.g. preferred_username) so self-registration can stamp it on registry
entries — including redeploys and startup restores, where no caller token
is available. Existing rows backfill to "".

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
        "definitions", sa.Column("owner_name", sa.String(255), nullable=False, server_default="")
    )


def downgrade() -> None:
    op.drop_column("definitions", "owner_name")
