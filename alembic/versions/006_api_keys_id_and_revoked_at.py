"""Add id PK and revoked_at to ops.api_keys

Revision ID: 006
Revises: 005
Create Date: 2026-06-30
"""

import sqlalchemy as sa
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("id", sa.Integer, autoincrement=True, nullable=True), schema="ops")
    op.execute("UPDATE ops.api_keys SET id = nextval(pg_get_serial_sequence('ops.api_keys', 'id'))")
    op.alter_column("api_keys", "id", nullable=False, schema="ops")
    op.drop_constraint("api_keys_pkey", "api_keys", schema="ops", type_="primary")
    op.create_primary_key("api_keys_pkey", "api_keys", ["id"], schema="ops")
    op.add_column("api_keys", sa.Column("revoked_at", sa.TIMESTAMP(timezone=True), nullable=True), schema="ops")


def downgrade() -> None:
    op.drop_column("api_keys", "revoked_at", schema="ops")
    op.drop_constraint("api_keys_pkey", "api_keys", schema="ops", type_="primary")
    op.create_primary_key("api_keys_pkey", "api_keys", ["key_hash"], schema="ops")
    op.drop_column("api_keys", "id", schema="ops")
