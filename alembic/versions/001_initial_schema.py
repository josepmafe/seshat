"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-07-01
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.schema.CreateSchema("ops", if_not_exists=True))
    op.execute(sa.schema.CreateSchema("knowledge_base", if_not_exists=True))
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("key_hash", sa.Text, nullable=False),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("revoked_at", sa.TIMESTAMP(timezone=True)),
        schema="ops",
    )
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("idempotency_key", sa.Text, unique=True),
        sa.Column("source_type", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("meeting_date", sa.Date, nullable=False),
        sa.Column("submission", sa.JSON, nullable=False),
        sa.Column("raw_blob_key", sa.Text, nullable=False),
        sa.Column("content_hash", sa.Text),
        sa.Column("error_payload", JSONB),
        sa.Column("mlflow_run_id", sa.Text),
        schema="ops",
    )
    op.create_index("ops_jobs_content_hash_idx", "jobs", ["content_hash"], schema="ops")
    op.create_table(
        "kb_nodes",
        sa.Column("node_id", sa.UUID, primary_key=True),
        sa.Column("schema_version", sa.Text, nullable=False, server_default="1.0"),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("quote_anchors", JSONB, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("state", sa.Text, nullable=False, server_default="current"),
        sa.Column("metadata", JSONB, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        schema="knowledge_base",
    )
    op.create_index("ix_kb_nodes_type", "kb_nodes", ["type"], schema="knowledge_base")
    op.create_index("ix_kb_nodes_state", "kb_nodes", ["state"], schema="knowledge_base")
    op.create_index(
        "ix_kb_nodes_metadata_job_id",
        "kb_nodes",
        [sa.text("(metadata->>'job_id')")],
        schema="knowledge_base",
    )
    op.create_index(
        "ix_kb_nodes_metadata_meeting_date",
        "kb_nodes",
        [sa.text("(metadata->>'meeting_date')")],
        schema="knowledge_base",
    )
    op.create_table(
        "kb_relationships",
        sa.Column("source_id", sa.UUID, sa.ForeignKey("knowledge_base.kb_nodes.node_id"), nullable=False),
        sa.Column("target_id", sa.UUID, sa.ForeignKey("knowledge_base.kb_nodes.node_id"), nullable=False),
        sa.Column("rel_type", sa.Text, nullable=False),
        sa.Column("job_id", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("source_id", "target_id", "rel_type"),
        schema="knowledge_base",
    )
    op.create_index("ix_kb_relationships_target_id", "kb_relationships", ["target_id"], schema="knowledge_base")


def downgrade() -> None:
    op.drop_index("ix_kb_relationships_target_id", table_name="kb_relationships", schema="knowledge_base")
    op.drop_table("kb_relationships", schema="knowledge_base")
    op.drop_index("ix_kb_nodes_metadata_meeting_date", table_name="kb_nodes", schema="knowledge_base")
    op.drop_index("ix_kb_nodes_metadata_job_id", table_name="kb_nodes", schema="knowledge_base")
    op.drop_index("ix_kb_nodes_state", table_name="kb_nodes", schema="knowledge_base")
    op.drop_index("ix_kb_nodes_type", table_name="kb_nodes", schema="knowledge_base")
    op.drop_table("kb_nodes", schema="knowledge_base")
    op.execute(sa.schema.DropSchema("knowledge_base", if_exists=True))
    op.drop_index("ops_jobs_content_hash_idx", table_name="jobs", schema="ops")
    op.drop_table("jobs", schema="ops")
    op.drop_table("api_keys", schema="ops")
    op.execute(sa.schema.DropSchema("ops", if_exists=True))
