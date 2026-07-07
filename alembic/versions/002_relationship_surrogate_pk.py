"""Add rel_id surrogate PK and source column to kb_relationships

Revision ID: 002
Revises: 001
Create Date: 2026-07-06
"""

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Drop composite PK, add rel_id UUID surrogate PK
    op.drop_constraint("kb_relationships_pkey", "kb_relationships", schema="knowledge_base")
    op.add_column(
        "kb_relationships",
        sa.Column("rel_id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        schema="knowledge_base",
    )
    op.create_primary_key(
        "kb_relationships_pkey",
        "kb_relationships",
        ["rel_id"],
        schema="knowledge_base",
    )
    op.create_unique_constraint(
        "uq_kb_relationships_edge",
        "kb_relationships",
        ["source_id", "target_id", "rel_type"],
        schema="knowledge_base",
    )

    # 2. Add source column
    op.add_column(
        "kb_relationships",
        sa.Column("source", sa.Text(), nullable=False, server_default="pipeline"),
        schema="knowledge_base",
    )


def downgrade() -> None:
    op.drop_column("kb_relationships", "source", schema="knowledge_base")

    op.drop_constraint("uq_kb_relationships_edge", "kb_relationships", schema="knowledge_base")
    op.drop_constraint("kb_relationships_pkey", "kb_relationships", schema="knowledge_base")
    op.drop_column("kb_relationships", "rel_id", schema="knowledge_base")
    op.create_primary_key(
        "kb_relationships_pkey",
        "kb_relationships",
        ["source_id", "target_id", "rel_type"],
        schema="knowledge_base",
    )
