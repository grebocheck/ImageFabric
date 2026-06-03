"""Read access to generated images."""

from __future__ import annotations

from sqlalchemy import String, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Image


async def list_images(
    session: AsyncSession, *, limit: int = 100, offset: int = 0, q: str | None = None
) -> list[Image]:
    stmt = select(Image)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(
            Image.id.ilike(like),
            Image.job_id.ilike(like),
            cast(Image.seed, String).ilike(like),
            cast(Image.params, String).ilike(like),
        ))
    stmt = stmt.order_by(Image.created_at.desc()).limit(limit).offset(offset)
    return list((await session.execute(stmt)).scalars().all())


async def get_image(session: AsyncSession, image_id: str) -> Image | None:
    return await session.get(Image, image_id)


def to_out_dict(img: Image) -> dict:
    return {
        "id": img.id,
        "job_id": img.job_id,
        "seed": img.seed,
        "width": img.width,
        "height": img.height,
        "params": img.params,
        "created_at": img.created_at,
        "url": f"/api/images/{img.id}/file",
        "thumb_url": f"/api/images/{img.id}/thumb" if img.thumb_path else None,
    }
