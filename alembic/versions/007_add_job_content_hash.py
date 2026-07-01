"""Add content_hash column to ops.jobs

Revision ID: 007
Revises: 006
Create Date: 2026-07-01
"""

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("content_hash", sa.Text(), nullable=True), schema="ops")
    op.create_index("ops_jobs_content_hash_idx", "jobs", ["content_hash"], schema="ops")


def downgrade() -> None:
    op.drop_index("ops_jobs_content_hash_idx", "jobs", schema="ops")
    op.drop_column("jobs", "content_hash", schema="ops")
