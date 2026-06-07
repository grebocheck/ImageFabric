"""Gallery service — History filters (P9.2) + tag/lora normalization.

Pure helpers run without a DB; the filter/stats coverage seeds rows in the
throwaway SQLite DB pinned by conftest and exercises the real SQL.
"""

from __future__ import annotations

import pytest

from app.core.enums import JobStatus, JobType
from app.db.models import Image, Job
from app.db.session import init_db, session_scope
from app.services import gallery_service as gs

# --------------------------------------------------------------- pure helpers


def test_normalize_tags_trims_dedups_and_caps():
    out = gs._normalize_tags(["  Hello   World ", "hello world", "", "x" * 60])
    assert out[0] == "Hello World"          # internal whitespace collapsed
    assert out.count("Hello World") == 1     # case-insensitive de-dup
    assert len(out[-1]) == 40                # 40-char cap


def test_normalize_tags_limit_of_32():
    out = gs._normalize_tags([f"t{i}" for i in range(50)])
    assert len(out) == 32


def test_lora_entries_filters_garbage_and_defaults_name():
    entries = gs._lora_entries({"loras": [
        {"id": "a", "name": "Anime"},
        {"id": "b"},            # missing name -> defaults to id
        {"name": "no-id"},      # dropped
        "nope",                 # dropped
    ]})
    assert entries == [{"id": "a", "name": "Anime"}, {"id": "b", "name": "b"}]


def test_to_out_dict_family_fallback_and_thumb():
    img = Image(id="i1", job_id="j1", path="/p.png", thumb_path=None,
                width=512, height=512, family=None, params={"family": "sdxl"}, tags=[])
    out = gs.to_out_dict(img)
    assert out["family"] == "sdxl"          # falls back to params
    assert out["thumb_url"] is None         # no thumb path
    assert out["url"] == "/api/images/i1/file"


# ------------------------------------------------------------- DB-backed tests


@pytest.fixture
async def seeded():
    await init_db()
    # The temp DB is shared across the test session; isolate from any rows other
    # tests (e.g. the stub integration run) may have left behind.
    from sqlalchemy import delete
    async with session_scope() as s:
        await s.execute(delete(Image))
        await s.execute(delete(Job))
    async with session_scope() as s:
        s.add(Job(id="job1", type=JobType.IMAGE, model_id="sdxl", status=JobStatus.DONE))
        s.add_all([
            Image(id="sq", job_id="job1", path="/sq.png", width=512, height=512,
                  family="sdxl", favorite=True, tags=["portrait", "fav"],
                  params={"model": "SDXL", "family": "sdxl"}),
            Image(id="land", job_id="job1", path="/l.png", width=1024, height=576,
                  family="flux", favorite=False, tags=["wide"],
                  params={"model": "FLUX", "family": "flux"}),
            Image(id="port", job_id="job1", path="/p.png", width=768, height=1024,
                  family="sdxl", favorite=False, tags=[],
                  params={"model": "SDXL", "family": "sdxl",
                          "loras": [{"id": "lora-x", "name": "X"}]}),
        ])
    yield
    from sqlalchemy import delete
    async with session_scope() as s:
        await s.execute(delete(Image))
        await s.execute(delete(Job))


async def _ids(**filters) -> set[str]:
    async with session_scope() as s:
        rows = await gs.list_images(s, **filters)
    return {r.id for r in rows}


async def test_size_filters(seeded):
    assert await _ids(size="square") == {"sq"}
    assert await _ids(size="landscape") == {"land"}
    assert await _ids(size="portrait") == {"port"}
    assert await _ids(size="large") == {"land", "port"}   # a dim >= 1024
    assert await _ids(size="small") == {"sq"}


async def test_family_and_favorite_and_model_filters(seeded):
    assert await _ids(family="sdxl") == {"sq", "port"}
    assert await _ids(family="flux") == {"land"}
    assert await _ids(favorite=True) == {"sq"}
    assert await _ids(model="FLUX") == {"land"}


async def test_tag_and_lora_filters(seeded):
    assert await _ids(tag="fav") == {"sq"}
    assert await _ids(lora="lora-x") == {"port"}


async def test_free_text_search(seeded):
    assert await _ids(q="lora-x") == {"port"}   # matches inside params JSON
    assert await _ids(q="wide") == {"land"}     # matches inside tags


async def test_stats_counts(seeded):
    async with session_scope() as s:
        st = await gs.stats(s)
    assert st["total"] == 3
    assert {m["model"]: m["count"] for m in st["by_model"]} == {"SDXL": 2, "FLUX": 1}
    assert {f["family"]: f["count"] for f in st["by_family"]} == {"sdxl": 2, "flux": 1}
    assert {t["tag"]: t["count"] for t in st["by_tag"]}["portrait"] == 1
    assert st["by_lora"] == [{"id": "lora-x", "name": "X", "count": 1}]
