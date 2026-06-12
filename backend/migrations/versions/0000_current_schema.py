"""current schema"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0000_current_schema"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _columns(table: str) -> set[str]:
    if not _table_exists(table):
        return set()
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}


def _index_exists(table: str, name: str) -> bool:
    if not _table_exists(table):
        return False
    return any(index["name"] == name for index in sa.inspect(op.get_bind()).get_indexes(table))


def _create_index_if_missing(name: str, table: str, columns: list[str]) -> None:
    if not _table_exists(table):
        return
    existing_columns = _columns(table)
    if not all(column in existing_columns for column in columns):
        return
    if not _index_exists(table, name):
        op.create_index(name, table, columns)


def upgrade() -> None:
    if not _table_exists("conversations"):
        op.create_table(
            "conversations",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("title", sa.String(length=200), nullable=False),
            sa.Column("model_id", sa.String(length=128), nullable=True),
            sa.Column("system", sa.Text(), nullable=True),
            sa.Column("params", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("jobs"):
        op.create_table(
            "jobs",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("type", sa.String(length=16), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("priority", sa.Integer(), nullable=False),
            sa.Column("model_id", sa.String(length=128), nullable=False),
            sa.Column("params", sa.JSON(), nullable=False),
            sa.Column("progress", sa.Float(), nullable=False),
            sa.Column("result", sa.JSON(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("model_profiles"):
        op.create_table(
            "model_profiles",
            sa.Column("model_id", sa.String(length=128), nullable=False),
            sa.Column("family", sa.String(length=16), nullable=False),
            sa.Column("quant", sa.String(length=32), nullable=True),
            sa.Column("ram_gb", sa.Float(), nullable=True),
            sa.Column("vram_gb", sa.Float(), nullable=True),
            sa.Column("samples", sa.Integer(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("model_id"),
        )

    if not _table_exists("notes"):
        op.create_table(
            "notes",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("title", sa.String(length=200), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("presets"):
        op.create_table(
            "presets",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("type", sa.String(length=16), nullable=False),
            sa.Column("params", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name"),
        )

    if not _table_exists("rag_documents"):
        op.create_table(
            "rag_documents",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("title", sa.String(length=240), nullable=False),
            sa.Column("source", sa.String(length=512), nullable=True),
            sa.Column("model_id", sa.String(length=128), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("images"):
        op.create_table(
            "images",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("job_id", sa.String(length=32), nullable=False),
            sa.Column("path", sa.String(length=512), nullable=False),
            sa.Column("thumb_path", sa.String(length=512), nullable=True),
            sa.Column("seed", sa.Integer(), nullable=True),
            sa.Column("width", sa.Integer(), nullable=True),
            sa.Column("height", sa.Integer(), nullable=True),
            sa.Column("family", sa.String(length=16), nullable=True),
            sa.Column("favorite", sa.Boolean(), nullable=False),
            sa.Column("tags", sa.JSON(), nullable=False),
            sa.Column("params", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("conversation_id", sa.String(length=32), nullable=False),
            sa.Column("role", sa.String(length=16), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("job_id", sa.String(length=32), nullable=True),
            sa.Column("error", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("rag_chunks"):
        op.create_table(
            "rag_chunks",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("document_id", sa.String(length=32), nullable=False),
            sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("embedding", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["document_id"], ["rag_documents.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    _create_index_if_missing("ix_conversations_updated_at", "conversations", ["updated_at"])
    _create_index_if_missing("ix_jobs_status", "jobs", ["status"])
    _create_index_if_missing("ix_jobs_priority", "jobs", ["priority"])
    _create_index_if_missing("ix_jobs_type", "jobs", ["type"])
    _create_index_if_missing("ix_jobs_created_at", "jobs", ["created_at"])
    _create_index_if_missing("ix_notes_updated_at", "notes", ["updated_at"])
    _create_index_if_missing("ix_notes_created_at", "notes", ["created_at"])
    _create_index_if_missing("ix_rag_documents_updated_at", "rag_documents", ["updated_at"])
    _create_index_if_missing("ix_rag_documents_created_at", "rag_documents", ["created_at"])
    _create_index_if_missing("ix_images_job_id", "images", ["job_id"])
    _create_index_if_missing("ix_images_family", "images", ["family"])
    _create_index_if_missing("ix_images_created_at", "images", ["created_at"])
    _create_index_if_missing("ix_messages_created_at", "messages", ["created_at"])
    _create_index_if_missing("ix_messages_conversation_id", "messages", ["conversation_id"])
    _create_index_if_missing("ix_rag_chunks_document_id", "rag_chunks", ["document_id"])


def downgrade() -> None:
    for table in (
        "rag_chunks",
        "messages",
        "images",
        "rag_documents",
        "presets",
        "notes",
        "model_profiles",
        "jobs",
        "conversations",
    ):
        if _table_exists(table):
            op.drop_table(table)
