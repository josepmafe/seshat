"""Add finished_at column to ops.jobs

Revision ID: 005
Revises: 004
Create Date: 2026-06-30
"""

import sqlalchemy as sa
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True), schema="ops")


def downgrade() -> None:
    op.drop_column("jobs", "finished_at", schema="ops")
