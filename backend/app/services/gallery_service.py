"""Read/maintenance access to generated images."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import String, and_, cast, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Image

# JSON path into the persisted param snapshot. Older rows may not have dedicated
# columns for these values, so the service keeps fallbacks for compatibility.
_MODEL_EXPR = Image.params["model"].as_string()
_FAMILY_EXPR = Image.params["family"].as_string()


def _family_expr():
    return func.coalesce(Image.family, _FAMILY_EXPR, "unknown")


def _apply_filters(stmt, *, q, model, family, size, lora, favorite, tag, date_from, date_to):
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(
            Image.id.ilike(like),
            Image.job_id.ilike(like),
            cast(Image.seed, String).ilike(like),
            cast(Image.tags, String).ilike(like),
            cast(Image.params, String).ilike(like),
        ))
    if model:
        stmt = stmt.where(_MODEL_EXPR == model)
    if family:
        stmt = stmt.where(_family_expr() == family)
    if size == "square":
        stmt = stmt.where(Image.width == Image.height)
    elif size == "landscape":
        stmt = stmt.where(Image.width > Image.height)
    elif size == "portrait":
        stmt = stmt.where(Image.height > Image.width)
    elif size == "large":
        stmt = stmt.where(or_(Image.width >= 1024, Image.height >= 1024))
    elif size == "small":
        stmt = stmt.where(and_(Image.width < 1024, Image.height < 1024))
    if lora:
        lora_item = func.json_each(Image.params, "$.loras").table_valued("value").alias("lora_item")
        stmt = stmt.where(
            exists(
                select(1)
                .select_from(lora_item)
                .where(func.json_extract(lora_item.c.value, "$.id") == lora.strip())
            )
        )
    if favorite is not None:
        stmt = stmt.where(Image.favorite.is_(favorite))
    if tag:
        tag_item = func.json_each(Image.tags).table_valued("value").alias("tag_item")
        stmt = stmt.where(
            exists(
                select(1)
                .select_from(tag_item)
                .where(tag_item.c.value == tag.strip())
            )
        )
    if date_from is not None:
        stmt = stmt.where(Image.created_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(Image.created_at <= date_to)
    return stmt


async def list_images(
    session: AsyncSession,
    *,
    limit: int = 100,
    offset: int = 0,
    q: str | None = None,
    model: str | None = None,
    family: str | None = None,
    size: str | None = None,
    lora: str | None = None,
    favorite: bool | None = None,
    tag: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[Image]:
    stmt = _apply_filters(
        select(Image),
        q=q,
        model=model,
        family=family,
        size=size,
        lora=lora,
        favorite=favorite,
        tag=tag,
        date_from=date_from,
        date_to=date_to,
    )
    stmt = stmt.order_by(Image.created_at.desc()).limit(limit).offset(offset)
    return list((await session.execute(stmt)).scalars().all())


async def get_image(session: AsyncSession, image_id: str) -> Image | None:
    return await session.get(Image, image_id)


async def get_images(session: AsyncSession, image_ids: list[str]) -> list[Image]:
    """Fetch images by id while preserving the caller's order and de-duping."""
    ids = list(dict.fromkeys(image_ids))
    if not ids:
        return []
    rows = list((await session.execute(select(Image).where(Image.id.in_(ids)))).scalars().all())
    by_id = {img.id: img for img in rows}
    return [by_id[image_id] for image_id in ids if image_id in by_id]


async def update_image(
    session: AsyncSession,
    image_id: str,
    *,
    favorite: bool | None = None,
    tags: list[str] | None = None,
) -> Image | None:
    img = await session.get(Image, image_id)
    if img is None:
        return None
    if favorite is not None:
        img.favorite = favorite
    if tags is not None:
        img.tags = _normalize_tags(tags)
    return img


async def delete_image(session: AsyncSession, image_id: str) -> bool:
    """Remove an image row and its files (best-effort on the filesystem)."""
    img = await session.get(Image, image_id)
    if img is None:
        return False
    for raw in (img.path, img.thumb_path):
        if not raw:
            continue
        try:
            Path(raw).unlink(missing_ok=True)
        except OSError:
            pass  # a locked/missing file should not block deleting the row
    await session.delete(img)
    return True


async def stats(session: AsyncSession) -> dict:
    """Generation counters for the History header: total, today, per-model."""
    total = (await session.execute(select(func.count(Image.id)))).scalar_one()

    start_of_day = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    today = (await session.execute(
        select(func.count(Image.id)).where(Image.created_at >= start_of_day)
    )).scalar_one()

    rows = (await session.execute(
        select(_MODEL_EXPR.label("model"), func.count(Image.id))
        .group_by(_MODEL_EXPR)
        .order_by(func.count(Image.id).desc())
    )).all()
    by_model = [{"model": name or "unknown", "count": count} for name, count in rows]

    family_expr = _family_expr()
    family_rows = (await session.execute(
        select(family_expr.label("family"), func.count(Image.id))
        .group_by(family_expr)
        .order_by(func.count(Image.id).desc())
    )).all()
    by_family = [{"family": name or "unknown", "count": count} for name, count in family_rows]

    lora_counts: Counter[tuple[str, str]] = Counter()
    for (params,) in (await session.execute(select(Image.params))).all():
        for item in _lora_entries(params):
            lora_counts[(item["id"], item["name"])] += 1
    by_lora = [
        {"id": lora_id, "name": name, "count": count}
        for (lora_id, name), count in lora_counts.most_common()
    ]

    tag_counts: Counter[str] = Counter()
    for (tags,) in (await session.execute(select(Image.tags))).all():
        for tag in _tag_entries(tags):
            tag_counts[tag] += 1
    by_tag = [{"tag": tag, "count": count} for tag, count in tag_counts.most_common()]

    return {
        "total": total,
        "today": today,
        "by_model": by_model,
        "by_family": by_family,
        "by_lora": by_lora,
        "by_tag": by_tag,
    }


def _lora_entries(params: Any) -> list[dict[str, str]]:
    if not isinstance(params, dict):
        return []
    raw = params.get("loras")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        lora_id = item.get("id")
        if not isinstance(lora_id, str) or not lora_id:
            continue
        name = item.get("name")
        out.append({"id": lora_id, "name": name if isinstance(name, str) and name else lora_id})
    return out


def _tag_entries(tags: Any) -> list[str]:
    if not isinstance(tags, list):
        return []
    return _normalize_tags([tag for tag in tags if isinstance(tag, str)])


def _normalize_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = " ".join(raw.strip().split())[:40]
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
        if len(out) >= 32:
            break
    return out


def to_out_dict(img: Image) -> dict:
    return {
        "id": img.id,
        "job_id": img.job_id,
        "seed": img.seed,
        "width": img.width,
        "height": img.height,
        "family": img.family or (img.params or {}).get("family") or "unknown",
        "favorite": bool(img.favorite),
        "tags": _tag_entries(img.tags),
        "params": img.params,
        "created_at": img.created_at,
        "url": f"/api/images/{img.id}/file",
        "thumb_url": f"/api/images/{img.id}/thumb" if img.thumb_path else None,
    }
