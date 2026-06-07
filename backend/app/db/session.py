"""Async SQLite engine + session helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import settings
from .models import Base

engine = create_async_engine(settings.db_url, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    settings.ensure_dirs()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_image_columns(conn)


async def _ensure_image_columns(conn) -> None:
    """Tiny SQLite migration for columns added after the original create_all."""
    rows = (await conn.execute(text("PRAGMA table_info(images)"))).all()
    columns = {row[1] for row in rows}
    if "family" not in columns:
        await conn.execute(text("ALTER TABLE images ADD COLUMN family TEXT"))
    if "favorite" not in columns:
        await conn.execute(text("ALTER TABLE images ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0"))
    if "tags" not in columns:
        await conn.execute(text("ALTER TABLE images ADD COLUMN tags JSON NOT NULL DEFAULT '[]'"))


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session for use in services and the worker loop."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with session_scope() as session:
        yield session
