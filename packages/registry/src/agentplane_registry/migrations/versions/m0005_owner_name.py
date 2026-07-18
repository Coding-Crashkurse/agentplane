"""Add the ``owner_name`` display column to ``entries``.

Introduced after 0.0.5-dev: ``owner`` (the OIDC subject) stays the
authorization key; ``owner_name`` carries the human-readable name
(e.g. preferred_username), denormalized at registration time. Existing rows
backfill to "" — the UI falls back to the subject.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "entries", sa.Column("owner_name", sa.String(255), nullable=False, server_default="")
    )


def downgrade() -> None:
    op.drop_column("entries", "owner_name")
