"""Repository code-context helpers for the Code workspace."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import settings

router = APIRouter(prefix="/api/code", tags=["code"])

IGNORED_DIRS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "bin",
    "data",
    "dist",
    "models",
    "node_modules",
}

TEXT_EXTS = {
    "",
    ".bat",
    ".cfg",
    ".css",
    ".env",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

MAX_FILE_BYTES = 180_000


def _root() -> Path:
    return settings.root.resolve()


def _rel(path: Path) -> str:
    return path.relative_to(_root()).as_posix()


def _inside_root(rel_path: str) -> Path:
    root = _root()
    path = (root / rel_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(400, "path escapes repository root")
    if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
        raise HTTPException(400, "path is ignored")
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return path


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTS or path.name in TEXT_EXTS


@router.get("/files")
async def list_code_files(
    q: str | None = Query(None, max_length=200),
    limit: int = Query(120, ge=1, le=500),
) -> list[dict]:
    query = (q or "").strip().lower()
    out: list[dict] = []
    root = _root()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and d != "dist"
        ]
        base = Path(dirpath)
        for name in sorted(filenames):
            path = base / name
            if not _is_text_candidate(path):
                continue
            rel = _rel(path)
            if query and query not in rel.lower():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            out.append({"path": rel, "size_bytes": size})
            if len(out) >= limit:
                return out
    return out


@router.get("/file")
async def get_code_file(path: str = Query(..., max_length=500)) -> dict:
    file_path = _inside_root(path)
    size = file_path.stat().st_size
    with file_path.open("rb") as f:
        data = f.read(MAX_FILE_BYTES + 1)
    truncated = len(data) > MAX_FILE_BYTES
    if b"\0" in data[:4096]:
        raise HTTPException(415, "file looks binary")
    content = data[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
    return {
        "path": _rel(file_path),
        "size_bytes": size,
        "content": content,
        "truncated": truncated,
    }
