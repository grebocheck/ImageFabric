from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3


def test_backup_main_writes_db_snapshot_and_outputs_manifest(tmp_path):
    backup = _load_backup_module()
    db_path = tmp_path / "hfabric.db"
    outputs = tmp_path / "outputs"
    out_dir = tmp_path / "backups"
    _create_db(db_path)
    image = outputs / "2026-01-01" / "img.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")

    rc = backup.main([
        "--db", str(db_path),
        "--outputs", str(outputs),
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    backup_dirs = list(out_dir.iterdir())
    assert len(backup_dirs) == 1
    with sqlite3.connect(backup_dirs[0] / "hfabric.db") as conn:
        assert conn.execute("SELECT value FROM sample").fetchone()[0] == "ok"
    manifest = json.loads((backup_dirs[0] / "outputs-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == [{
        "path": "2026-01-01/img.png",
        "size": 3,
        "mtime": image.stat().st_mtime,
    }]


def test_backup_retention_keeps_newest_n(tmp_path):
    backup = _load_backup_module()
    db_path = tmp_path / "hfabric.db"
    outputs = tmp_path / "outputs"
    out_dir = tmp_path / "backups"
    _create_db(db_path)
    outputs.mkdir()

    for _ in range(3):
        assert backup.main([
            "--db", str(db_path),
            "--outputs", str(outputs),
            "--out-dir", str(out_dir),
            "--keep", "2",
        ]) == 0

    backups = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("hfabric-")]
    assert len(backups) == 2


def _create_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sample (value) VALUES ('ok')")


def _load_backup_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "backup.py"
    spec = importlib.util.spec_from_file_location("hfabric_backup_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
