#!/usr/bin/env python3
"""Create an HFabric DB snapshot and outputs manifest."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "hfabric.db"
DEFAULT_OUTPUTS = ROOT / "data" / "outputs"
DEFAULT_OUT_DIR = ROOT / "data" / "backups"
BACKUP_PREFIX = "hfabric-"


def outputs_manifest(outputs_dir: Path) -> list[dict[str, object]]:
    if not outputs_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in outputs_dir.rglob("*") if p.is_file()):
        stat = path.stat()
        rows.append({
            "path": path.relative_to(outputs_dir).as_posix(),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        })
    return rows


def snapshot_db(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"database not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(src)) as source:
        with closing(sqlite3.connect(dest)) as target:
            source.backup(target)


def write_manifest(outputs_dir: Path, dest: Path) -> None:
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "outputs_dir": str(outputs_dir),
        "files": outputs_manifest(outputs_dir),
    }
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def create_backup(
    *,
    db_path: Path = DEFAULT_DB,
    outputs_dir: Path = DEFAULT_OUTPUTS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    base_name = f"{BACKUP_PREFIX}{stamp}"
    for suffix in range(100):
        name = base_name if suffix == 0 else f"{base_name}-{suffix:02d}"
        backup_dir = out_dir / name
        try:
            backup_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"could not create a unique backup directory for {base_name}")
    try:
        snapshot_db(db_path, backup_dir / "hfabric.db")
        write_manifest(outputs_dir, backup_dir / "outputs-manifest.json")
    except Exception:
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    return backup_dir


def apply_retention(out_dir: Path, keep: int | None) -> list[Path]:
    if keep is None or keep < 0 or not out_dir.exists():
        return []
    backups = sorted(
        [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(BACKUP_PREFIX)],
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    removed: list[Path] = []
    for path in backups[keep:]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite DB to snapshot")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS, help="outputs directory to manifest")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="directory for backup snapshots")
    parser.add_argument("--keep", type=int, default=None, help="keep only the newest N backups")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    backup_dir = create_backup(db_path=args.db, outputs_dir=args.outputs, out_dir=args.out_dir)
    removed = apply_retention(args.out_dir, args.keep)
    print(f"backup: {backup_dir}")
    if removed:
        print(f"removed: {len(removed)} old backup(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
