"""Gallery: list image metadata and serve the files/thumbnails."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from ..schemas import ImageExportIn, ImageOut
from ..services import gallery_service
from .deps import get_session

router = APIRouter(prefix="/api/images", tags=["gallery"])


@router.get("", response_model=list[ImageOut])
async def list_images(
    limit: int = Query(100, le=500),
    offset: int = 0,
    q: str | None = Query(None, max_length=200),
    model: str | None = Query(None, max_length=200),
    size: str | None = Query(None, pattern="^(square|landscape|portrait|large|small)$"),
    lora: str | None = Query(None, max_length=200),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[ImageOut]:
    images = await gallery_service.list_images(
        session, limit=limit, offset=offset, q=q,
        model=model, size=size, lora=lora, date_from=date_from, date_to=date_to,
    )
    return [ImageOut.model_validate(gallery_service.to_out_dict(i)) for i in images]


@router.get("/stats")
async def image_stats(session: AsyncSession = Depends(get_session)) -> dict:
    """Generation counters for the History header (total / today / per-model)."""
    return await gallery_service.stats(session)


@router.post("/export")
async def export_images(body: ImageExportIn, session: AsyncSession = Depends(get_session)) -> FileResponse:
    images = await gallery_service.get_images(session, body.image_ids)
    if not images:
        raise HTTPException(404, "no selected images found")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        zip_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for img in images:
                payload = ImageOut.model_validate(gallery_service.to_out_dict(img)).model_dump(mode="json")
                zf.writestr(
                    f"metadata/{img.id}.json",
                    json.dumps(payload, ensure_ascii=False, indent=2),
                )
                path = Path(img.path)
                if path.exists():
                    zf.write(path, f"images/{img.id}{path.suffix or '.png'}")
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"hfabric-images-{len(images)}.zip",
        background=BackgroundTask(lambda: zip_path.unlink(missing_ok=True)),
    )


@router.delete("/{image_id}")
async def delete_image(image_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    if not await gallery_service.delete_image(session, image_id):
        raise HTTPException(404, "image not found")
    return {"deleted": image_id}


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


@router.post("/{image_id}/reveal")
async def reveal_image(image_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Open the OS file manager with the image file selected. Local-app
    convenience — the browser cannot reach the desktop file manager itself."""
    img = await gallery_service.get_image(session, image_id)
    if not img or not Path(img.path).exists():
        raise HTTPException(404, "image not found")
    path = Path(img.path)
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
    except OSError as exc:  # pragma: no cover - desktop-only path
        raise HTTPException(500, f"could not open file manager: {exc}")
    return {"revealed": str(path)}


@router.get("/{image_id}/thumb")
async def image_thumb(image_id: str, session: AsyncSession = Depends(get_session)) -> FileResponse:
    img = await gallery_service.get_image(session, image_id)
    if not img or not img.thumb_path or not Path(img.thumb_path).exists():
        raise HTTPException(404, "thumbnail not found")
    return FileResponse(img.thumb_path, media_type="image/webp")
