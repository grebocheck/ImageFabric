"""Gallery: list image metadata and serve the files/thumbnails."""

from __future__ import annotations

from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from uuid import uuid4
import zipfile

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from ..config import settings
from ..schemas import ImageExportIn, ImageOut, ImageUpdateIn
from ..services import gallery_service
from ..util import uploads as uploads_util
from .deps import get_session

router = APIRouter(prefix="/api/images", tags=["gallery"])


@router.get("", response_model=list[ImageOut])
async def list_images(
    limit: int = Query(100, le=500),
    offset: int = 0,
    q: str | None = Query(None, max_length=200),
    model: str | None = Query(None, max_length=200),
    family: str | None = Query(None, pattern="^(flux|flux2|qwen-image|z-image|sdxl|unknown)$"),
    size: str | None = Query(None, pattern="^(square|landscape|portrait|large|small)$"),
    lora: str | None = Query(None, max_length=200),
    favorite: bool | None = None,
    tag: str | None = Query(None, max_length=40),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[ImageOut]:
    images = await gallery_service.list_images(
        session, limit=limit, offset=offset, q=q,
        model=model, family=family, size=size, lora=lora, favorite=favorite, tag=tag,
        date_from=date_from, date_to=date_to,
    )
    return [ImageOut.model_validate(gallery_service.to_out_dict(i)) for i in images]


@router.get("/stats")
async def image_stats(session: AsyncSession = Depends(get_session)) -> dict:
    """Generation counters for the History header (total / today / per-model)."""
    return await gallery_service.stats(session)


async def _read_upload_bytes(file: UploadFile) -> bytes:
    raw = await file.read()
    if len(raw) > settings.image_upload_max_mb * 1024 * 1024:
        raise HTTPException(413, f"image exceeds {settings.image_upload_max_mb} MB")
    return raw


@router.post("/upload")
async def upload_init_image(file: UploadFile = File(...)) -> dict:
    """Accept a source image for img2img/inpainting. We re-encode to PNG via PIL
    so the stored file is normalized, and return an opaque token the composer
    puts in a job's ``init_image`` param."""
    raw = await _read_upload_bytes(file)
    try:
        from PIL import Image as PILImage  # noqa: PLC0415

        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "unsupported or corrupt image")
    token = uuid4().hex
    img.save(uploads_util.uploads_dir() / f"{token}.png", format="PNG")
    return {"init_image": token, "url": f"/api/images/upload/{token}", "width": img.width, "height": img.height}


@router.post("/upload-mask")
async def upload_mask_image(file: UploadFile = File(...)) -> dict:
    """Accept an inpainting mask. White/bright pixels mark the repaint region;
    grey pixels are preserved for feathered edges."""
    raw = await _read_upload_bytes(file)
    try:
        from PIL import Image as PILImage
        from PIL import ImageChops  # noqa: PLC0415

        img = PILImage.open(io.BytesIO(raw)).convert("RGBA")
        grey = img.convert("L")
        alpha = img.getchannel("A")
        if alpha.getextrema() != (255, 255):
            grey = ImageChops.multiply(grey, alpha)
    except Exception:
        raise HTTPException(400, "unsupported or corrupt mask image")
    token = uuid4().hex
    grey.save(uploads_util.uploads_dir() / f"{token}.png", format="PNG")
    return {"mask_image": token, "url": f"/api/images/upload/{token}", "width": grey.width, "height": grey.height}


@router.get("/upload/{token}")
async def upload_file(token: str) -> FileResponse:
    path = uploads_util.resolve_upload(token)
    if path is None or not path.exists():
        raise HTTPException(404, "upload not found")
    return FileResponse(path, media_type="image/png")


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


@router.patch("/{image_id}", response_model=ImageOut)
async def update_image(image_id: str, body: ImageUpdateIn, session: AsyncSession = Depends(get_session)) -> ImageOut:
    img = await gallery_service.update_image(session, image_id, favorite=body.favorite, tags=body.tags)
    if img is None:
        raise HTTPException(404, "image not found")
    return ImageOut.model_validate(gallery_service.to_out_dict(img))


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
