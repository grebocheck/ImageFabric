"""Persistent notes/scratch workspace."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Note
from ..schemas import NoteCreate, NoteOut, NoteUpdate
from .deps import get_session

router = APIRouter(prefix="/api/notes", tags=["notes"])


def _now() -> datetime:
    return datetime.now(UTC)


def _title(value: str | None) -> str:
    title = (value or "").strip()
    return title[:200] if title else "Untitled note"


@router.get("", response_model=list[NoteOut])
async def list_notes(
    q: str | None = None,
    limit: int = Query(200, le=1000),
    session: AsyncSession = Depends(get_session),
) -> list[NoteOut]:
    stmt = select(Note).order_by(Note.updated_at.desc()).limit(limit)
    if q and q.strip():
        pattern = f"%{q.strip()}%"
        stmt = (
            select(Note)
            .where(or_(Note.title.ilike(pattern), Note.content.ilike(pattern)))
            .order_by(Note.updated_at.desc())
            .limit(limit)
        )
    rows = (await session.execute(stmt)).scalars().all()
    return [NoteOut.model_validate(n) for n in rows]


@router.post("", response_model=NoteOut)
async def create_note(
    body: NoteCreate, session: AsyncSession = Depends(get_session)
) -> NoteOut:
    note = Note(title=_title(body.title), content=body.content)
    session.add(note)
    await session.commit()
    return NoteOut.model_validate(note)


@router.get("/{note_id}", response_model=NoteOut)
async def get_note(note_id: str, session: AsyncSession = Depends(get_session)) -> NoteOut:
    note = await session.get(Note, note_id)
    if not note:
        raise HTTPException(404, "note not found")
    return NoteOut.model_validate(note)


@router.patch("/{note_id}", response_model=NoteOut)
async def update_note(
    note_id: str, body: NoteUpdate, session: AsyncSession = Depends(get_session)
) -> NoteOut:
    note = await session.get(Note, note_id)
    if not note:
        raise HTTPException(404, "note not found")
    if body.title is not None:
        note.title = _title(body.title)
    if body.content is not None:
        note.content = body.content
    note.updated_at = _now()
    await session.commit()
    return NoteOut.model_validate(note)


@router.delete("/{note_id}")
async def delete_note(note_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    note = await session.get(Note, note_id)
    if not note:
        raise HTTPException(404, "note not found")
    await session.delete(note)
    await session.commit()
    return {"deleted": note_id}
