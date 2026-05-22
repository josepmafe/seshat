"""Drop chunk_index from kb_nodes

Revision ID: 003
Revises: 002
Create Date: 2026-05-20
"""

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("kb_nodes", "chunk_index", schema="ops")


def downgrade() -> None:
    op.add_column(
        "kb_nodes",
        sa.Column("chunk_index", sa.Integer, nullable=True),
        schema="ops",
    )
