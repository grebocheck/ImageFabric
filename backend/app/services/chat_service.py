"""Chat persistence: conversations + messages. Pure DB ops; routers/worker
handle event emission and the actual LLM job."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Conversation, Message


def _now() -> datetime:
    return datetime.now(UTC)


async def create_conversation(
    session: AsyncSession,
    *,
    title: str | None = None,
    model_id: str | None = None,
    system: str | None = None,
    params: dict | None = None,
) -> Conversation:
    conv = Conversation(
        title=title or "New chat",
        model_id=model_id,
        system=system,
        params=params or {},
    )
    session.add(conv)
    await session.flush()
    return conv


async def list_conversations(session: AsyncSession, *, limit: int = 200) -> list[Conversation]:
    stmt = select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars().all())


async def get_conversation(session: AsyncSession, conv_id: str) -> Conversation | None:
    return await session.get(Conversation, conv_id)


async def get_messages(session: AsyncSession, conv_id: str) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at.asc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def update_conversation(
    session: AsyncSession, conv_id: str, **fields
) -> Conversation | None:
    conv = await session.get(Conversation, conv_id)
    if not conv:
        return None
    for key, value in fields.items():
        if value is not None and hasattr(conv, key):
            setattr(conv, key, value)
    conv.updated_at = _now()
    return conv


async def delete_conversation(session: AsyncSession, conv_id: str) -> bool:
    conv = await session.get(Conversation, conv_id)
    if not conv:
        return False
    await session.delete(conv)
    return True


async def add_message(
    session: AsyncSession,
    conv_id: str,
    *,
    role: str,
    content: str = "",
    job_id: str | None = None,
) -> Message:
    msg = Message(conversation_id=conv_id, role=role, content=content, job_id=job_id)
    session.add(msg)
    await session.flush()
    return msg


async def truncate_from(session: AsyncSession, conv_id: str, message_id: str) -> int:
    """Delete ``message_id`` and every message created after it (for edit /
    regenerate). Returns the number removed."""
    target = await session.get(Message, message_id)
    if not target or target.conversation_id != conv_id:
        return 0
    rows = (await session.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .where(Message.created_at >= target.created_at)
    )).scalars().all()
    for msg in rows:
        await session.delete(msg)
    return len(rows)


async def touch(session: AsyncSession, conv_id: str) -> None:
    conv = await session.get(Conversation, conv_id)
    if conv:
        conv.updated_at = _now()


async def finalize_assistant_message(
    session: AsyncSession, message_id: str, content: str, *, error: bool = False
) -> None:
    """Called by the worker when an LLM job finishes: write the reply into the
    placeholder assistant message and bump the conversation."""
    msg = await session.get(Message, message_id)
    if not msg:
        return
    msg.content = content
    msg.error = error
    msg.job_id = None
    await touch(session, msg.conversation_id)
