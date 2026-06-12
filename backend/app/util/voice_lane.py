"""Runtime marker for live voice sessions that park GPU jobs."""

from __future__ import annotations

from ..config import settings

MARKER = "voice-lane.active"


def marker_path():
    return settings.runtime_dir / MARKER


def set_active(active: bool) -> None:
    path = marker_path()
    if active:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1\n", encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def is_active() -> bool:
    return marker_path().exists()
