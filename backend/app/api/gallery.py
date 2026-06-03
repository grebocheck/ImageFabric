"""Gallery: list image metadata and serve the files/thumbnails."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import ImageOut
from ..services import gallery_service
from .deps import get_session

router = APIRouter(prefix="/api/images", tags=["gallery"])


@router.get("", response_model=list[ImageOut])
async def list_images(
    limit: int = Query(100, le=500),
    offset: int = 0,
    q: str | None = Query(None, max_length=200),
    session: AsyncSession = Depends(get_session),
) -> list[ImageOut]:
    images = await gallery_service.list_images(session, limit=limit, offset=offset, q=q)
    return [ImageOut.model_validate(gallery_service.to_out_dict(i)) for i in images]


@router.get("/{image_id}/file")
async def image_file(image_id: str, session: AsyncSession = Depends(get_session)) -> FileResponse:
    img = await gallery_service.get_image(session, image_id)
    if not img or not Path(img.path).exists():
        raise HTTPException(404, "image not found")
    return FileResponse(img.path, media_type="image/png")


@router.get("/{image_id}/metadata")
async def image_metadata(image_id: str, session: AsyncSession = Depends(get_session)) -> JSONResponse:
    img = await gallery_service.get_image(session, image_id)
    if not img:
        raise HTTPException(404, "image not found")
    payload = ImageOut.model_validate(gallery_service.to_out_dict(img)).model_dump(mode="json")
    return JSONResponse(
        payload,
        headers={"Content-Disposition": f'attachment; filename="{image_id}.metadata.json"'},
    )


@router.get("/{image_id}/thumb")
async def image_thumb(image_id: str, session: AsyncSession = Depends(get_session)) -> FileResponse:
    img = await gallery_service.get_image(session, image_id)
    if not img or not img.thumb_path or not Path(img.thumb_path).exists():
        raise HTTPException(404, "thumbnail not found")
    return FileResponse(img.thumb_path, media_type="image/webp")
