"""RAG document workspace endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.models import Note, RagChunk, RagDocument
from ..services.embedding_service import (
    embedding_service,
    list_embedding_models,
)
from ..services.rag_service import resolve_embedding_model_id
from ..services.rag_service import search_documents as run_rag_search
from .deps import get_session

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RagDocumentCreate(BaseModel):
    title: str | None = None
    content: str = Field(min_length=1, max_length=1_000_000)
    source: str | None = None
    model_id: str | None = None


class RagSearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    model_id: str | None = None


def _now() -> datetime:
    return datetime.now(UTC)


def _title(value: str | None, fallback: str = "Untitled document") -> str:
    clean = re.sub(r"\s+", " ", (value or "").strip())
    return clean[:240] if clean else fallback


def _chunk_text(text: str) -> list[str]:
    clean = re.sub(r"\r\n?", "\n", text).strip()
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    if not clean:
        return []

    target = max(400, settings.rag_chunk_chars)
    overlap = max(0, min(settings.rag_chunk_overlap, target // 2))
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + target)
        if end < len(clean):
            boundary = max(clean.rfind("\n\n", start, end), clean.rfind(". ", start, end))
            if boundary > start + target // 2:
                end = boundary + (1 if clean[boundary:boundary + 1] == "." else 0)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _doc_out(doc: RagDocument, chunks_count: int) -> dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "source": doc.source,
        "model_id": doc.model_id,
        "chunks_count": chunks_count,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }


@router.get("/status")
async def rag_status() -> dict:
    models = list_embedding_models()
    return {
        "binary": str(settings.llama_server_bin),
        "binary_exists": settings.llama_server_bin.exists(),
        "models_dir": str(settings.embed_models_dir),
        "models": models,
        "ready": settings.llama_server_bin.exists() and bool(models),
        "port": settings.llama_embed_port,
        "gpu_layers": settings.embed_gpu_layers,
        "chunk_chars": settings.rag_chunk_chars,
        "chunk_overlap": settings.rag_chunk_overlap,
    }


@router.get("/documents")
async def list_documents(
    q: str | None = Query(None, max_length=200),
    limit: int = Query(200, ge=1, le=1000),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    stmt = (
        select(RagDocument, func.count(RagChunk.id))
        .outerjoin(RagChunk)
        .group_by(RagDocument.id)
        .order_by(RagDocument.updated_at.desc())
        .limit(limit)
    )
    query = (q or "").strip()
    if query:
        pattern = f"%{query}%"
        stmt = (
            select(RagDocument, func.count(RagChunk.id))
            .outerjoin(RagChunk)
            .where((RagDocument.title.ilike(pattern)) | (RagDocument.source.ilike(pattern)))
            .group_by(RagDocument.id)
            .order_by(RagDocument.updated_at.desc())
            .limit(limit)
        )
    rows = (await session.execute(stmt)).all()
    return [_doc_out(doc, int(count)) for doc, count in rows]


@router.post("/documents")
async def create_document(
    body: RagDocumentCreate,
    session: AsyncSession = Depends(get_session),
) -> dict:
    return await _index_document(
        session,
        title=_title(body.title),
        content=body.content,
        source=body.source,
        model_id=body.model_id,
    )


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    model_id: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
) -> dict:
    payload = await file.read(1_000_001)
    if len(payload) > 1_000_000:
        raise HTTPException(413, "document upload exceeds 1 MB")
    if not payload:
        raise HTTPException(422, "document is empty")
    content = payload.decode("utf-8", errors="replace")
    return await _index_document(
        session,
        title=_title(title, fallback=file.filename or "Uploaded document"),
        content=content,
        source=file.filename,
        model_id=model_id,
    )


@router.post("/documents/from-note/{note_id}")
async def create_from_note(
    note_id: str,
    model_id: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    note = await session.get(Note, note_id)
    if not note:
        raise HTTPException(404, "note not found")
    return await _index_document(
        session,
        title=note.title,
        content=note.content,
        source=f"note:{note.id}",
        model_id=model_id,
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    doc = await session.get(RagDocument, document_id)
    if not doc:
        raise HTTPException(404, "document not found")
    await session.delete(doc)
    await session.commit()
    return {"deleted": document_id}


@router.post("/search")
async def search_documents(
    body: RagSearchIn,
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        return await run_rag_search(
            session,
            query=body.query,
            top_k=body.top_k,
            model_id=body.model_id,
        )
    except KeyError as exc:
        raise HTTPException(404, "embedding model not found") from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


async def _index_document(
    session: AsyncSession,
    *,
    title: str,
    content: str,
    source: str | None,
    model_id: str | None,
) -> dict:
    chunks = _chunk_text(content)
    if not chunks:
        raise HTTPException(422, "document has no indexable text")
    try:
        resolved_model_id = resolve_embedding_model_id(model_id)
    except KeyError as exc:
        raise HTTPException(404, "embedding model not found") from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    try:
        vectors = await embedding_service.embed(
            [f"search_document: {chunk}" for chunk in chunks],
            model_id=resolved_model_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"embedding failed: {exc}") from exc

    doc = RagDocument(
        title=title,
        source=source,
        model_id=resolved_model_id,
        updated_at=_now(),
    )
    session.add(doc)
    await session.flush()
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        session.add(RagChunk(
            document_id=doc.id,
            chunk_index=idx,
            text=chunk,
            embedding=vector,
        ))
    await session.commit()
    return _doc_out(doc, len(chunks))
