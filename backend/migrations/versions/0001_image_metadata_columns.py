"""add image metadata columns"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_image_metadata_columns"
down_revision = "0000_current_schema"
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


def upgrade() -> None:
    if not _table_exists("images"):
        return
    columns = _columns("images")
    with op.batch_alter_table("images") as batch_op:
        if "family" not in columns:
            batch_op.add_column(sa.Column("family", sa.String(length=16), nullable=True))
        if "favorite" not in columns:
            batch_op.add_column(
                sa.Column("favorite", sa.Boolean(), nullable=False, server_default=sa.false())
            )
        if "tags" not in columns:
            batch_op.add_column(sa.Column("tags", sa.JSON(), nullable=False, server_default="[]"))
    if "family" in _columns("images") and not _index_exists("images", "ix_images_family"):
        op.create_index("ix_images_family", "images", ["family"])


def downgrade() -> None:
    if not _table_exists("images"):
        return
    columns = _columns("images")
    with op.batch_alter_table("images") as batch_op:
        if "family" in columns:
            if _index_exists("images", "ix_images_family"):
                batch_op.drop_index("ix_images_family")
            batch_op.drop_column("family")
        if "favorite" in columns:
            batch_op.drop_column("favorite")
        if "tags" in columns:
            batch_op.drop_column("tags")
