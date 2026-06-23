"""Add ts_content tsvector column to langchain_pg_embedding

Revision ID: 004
Revises: 003
Create Date: 2026-06-22
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "langchain_pg_embedding",
        sa.Column(
            "ts_content",
            TSVECTOR,
            sa.Computed("to_tsvector('english', document)", persisted=True),
        ),
    )
    op.create_index(
        "langchain_pg_embedding_ts_content_gin",
        "langchain_pg_embedding",
        ["ts_content"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("langchain_pg_embedding_ts_content_gin", table_name="langchain_pg_embedding")
    op.drop_column("langchain_pg_embedding", "ts_content")
