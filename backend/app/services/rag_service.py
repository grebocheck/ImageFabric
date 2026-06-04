"""Shared RAG search helpers."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.models import RagChunk, RagDocument
from .embedding_service import embedding_model_map, embedding_service


def resolve_embedding_model_id(model_id: str | None = None) -> str:
    models = embedding_model_map()
    if not models:
        raise RuntimeError(f"no embedding models found in {settings.embed_models_dir}")
    if model_id:
        if model_id not in models:
            raise KeyError("embedding model not found")
        return model_id
    return next(iter(models))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


async def search_documents(
    session: AsyncSession,
    *,
    query: str,
    top_k: int = 5,
    model_id: str | None = None,
) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query:
        return {"query": clean_query, "results": [], "context": ""}

    rows = (await session.execute(
        select(RagChunk, RagDocument)
        .join(RagDocument, RagDocument.id == RagChunk.document_id)
    )).all()
    if not rows:
        return {"query": clean_query, "results": [], "context": ""}

    resolved_model_id = resolve_embedding_model_id(model_id)
    query_vec = (await embedding_service.embed(
        [f"search_query: {clean_query}"],
        model_id=resolved_model_id,
    ))[0]

    scored = []
    for chunk, doc in rows:
        if not chunk.embedding:
            continue
        score = _dot(query_vec, [float(x) for x in chunk.embedding])
        scored.append((score, chunk, doc))
    scored.sort(key=lambda item: item[0], reverse=True)

    limit = max(1, min(20, int(top_k)))
    results = [
        {
            "document_id": doc.id,
            "document_title": doc.title,
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "score": float(score),
        }
        for score, chunk, doc in scored[:limit]
    ]
    context = "\n\n".join(
        f"[{idx + 1}] {item['document_title']} (chunk {item['chunk_index'] + 1})\n{item['text']}"
        for idx, item in enumerate(results)
    )
    return {"query": clean_query, "results": results, "context": context}
