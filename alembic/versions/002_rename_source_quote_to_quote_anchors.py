"""Rename source_quote to quote_anchors and change type to JSONB

Revision ID: 002
Revises: 001
Create Date: 2026-05-14
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "kb_nodes",
        "source_quote",
        new_column_name="quote_anchors",
        existing_type=sa.Text,
        type_=JSONB,
        postgresql_using="'[]'::jsonb",
        schema="ops",
    )


def downgrade() -> None:
    op.alter_column(
        "kb_nodes",
        "quote_anchors",
        new_column_name="source_quote",
        existing_type=JSONB,
        type_=sa.Text,
        postgresql_using="quote_anchors::text",
        schema="ops",
    )
