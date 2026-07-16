"""Baseline: the runtime schema exactly as released in agentplane-runtime 0.0.2.

Deliberately NOT the current ``db.py`` metadata: 0.0.2 predates the ``group``
columns on ``definitions`` and ``resources`` (added by revision 0002), and a
real 0.0.2 database gets stamped at this revision, so the baseline must match
what 0.0.2's ``create_all`` produced.

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
        "definitions",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("draft_json", sa.String(), nullable=False),
        sa.Column("deployed_version", sa.Integer(), nullable=True),
        sa.Column("registry_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f("ix_definitions_owner"), "definitions", ["owner"])
    op.create_index(op.f("ix_definitions_status"), "definitions", ["status"])
    op.create_table(
        "definition_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("version_label", sa.String(64), nullable=True),
        sa.Column("definition_json", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", "version", name="uq_versions_name_version"),
        sa.UniqueConstraint("name", "version_label", name="uq_versions_name_label"),
    )
    op.create_index(op.f("ix_definition_versions_name"), "definition_versions", ["name"])
    op.create_table(
        "resources",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("config_json", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f("ix_resources_kind"), "resources", ["kind"])
    op.create_index(op.f("ix_resources_owner"), "resources", ["owner"])
    op.create_table(
        "secrets",
        sa.Column("ref", sa.String(255), primary_key=True),
        sa.Column("ciphertext", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("secrets")
    op.drop_index(op.f("ix_resources_owner"), table_name="resources")
    op.drop_index(op.f("ix_resources_kind"), table_name="resources")
    op.drop_table("resources")
    op.drop_index(op.f("ix_definition_versions_name"), table_name="definition_versions")
    op.drop_table("definition_versions")
    op.drop_index(op.f("ix_definitions_status"), table_name="definitions")
    op.drop_index(op.f("ix_definitions_owner"), table_name="definitions")
    op.drop_table("definitions")
