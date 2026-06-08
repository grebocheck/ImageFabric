"""Fetch Qwen-Image-2512 and Z-Image-Turbo Diffusers repos.

These are public, multi-file text-to-image model repos. They are intentionally
kept out of ``fetch_models.py`` because together they are roughly 84 GB.

The Hugging Face snapshot/Xet path can stall on some Windows networks, so this
script downloads each file with ``curl`` using resume + retry.

    python scripts/fetch_qwen_z_image.py
"""

from __future__ import annotations

import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "models" / "image"

JOBS = [
    ("Tongyi-MAI/Z-Image-Turbo", IMAGE / "z-image-turbo", ["assets/*", ".gitattributes"]),
    ("Qwen/Qwen-Image-2512", IMAGE / "qwen-image-2512", [".gitattributes"]),
]
MAX_WORKERS = 5
STALE_SECONDS = 120
MAX_ATTEMPTS = 80


def _dir_size_gb(path: Path) -> float:
    total = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return round(total / 1024**3, 2)


def _wanted_files(repo: str, ignore: list[str]) -> list[tuple[str, int | None]]:
    info = HfApi().model_info(repo, files_metadata=True)
    files: list[tuple[str, int | None]] = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if any(fnmatch.fnmatch(name, pattern) for pattern in ignore):
            continue
        files.append((name, getattr(sibling, "size", None)))
    return sorted(files)


def _curl_bin() -> str:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl.exe/curl not found on PATH")
    return curl


def _is_complete(path: Path, expected_size: int | None) -> bool:
    return bool(expected_size and path.is_file() and path.stat().st_size == expected_size)


def _download_file(curl: str, repo: str, filename: str, dest: Path, expected_size: int | None) -> None:
    out = dest / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    if _is_complete(out, expected_size):
        print(f"[skip]  {filename}", flush=True)
        return

    url = f"https://huggingface.co/{repo}/resolve/main/{quote(filename)}"
    size = f" ({round(expected_size / 1024**3, 2)} GB)" if expected_size else ""
    print(f"[file]  {filename}{size}", flush=True)
    args = [
        curl,
        "--location",
        "--fail",
        "--continue-at",
        "-",
        "--retry",
        "4",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-time",
        "60",
        "--speed-limit",
        "1024",
        "--output",
        str(out),
        url,
    ]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if _is_complete(out, expected_size):
            return
        print(f"[try]   {filename} attempt {attempt}", flush=True)
        if _run_curl_with_watchdog(args, out):
            if not expected_size or _is_complete(out, expected_size):
                return
        time.sleep(5)
    raise RuntimeError(f"Could not finish {filename} after {MAX_ATTEMPTS} attempts")


def _run_curl_with_watchdog(args: list[str], out: Path) -> bool:
    proc = subprocess.Popen(args)
    last_size = out.stat().st_size if out.is_file() else 0
    last_change = time.monotonic()
    while proc.poll() is None:
        time.sleep(10)
        size = out.stat().st_size if out.is_file() else 0
        if size != last_size:
            last_size = size
            last_change = time.monotonic()
            continue
        if time.monotonic() - last_change >= STALE_SECONDS:
            print(f"[stale] restarting curl at {round(size / 1024**3, 2)} GB", flush=True)
            proc.kill()
            proc.wait()
            return False
    return proc.returncode == 0


def main() -> None:
    IMAGE.mkdir(parents=True, exist_ok=True)
    curl = _curl_bin()
    for repo, dest, ignore in JOBS:
        dest.mkdir(parents=True, exist_ok=True)
        files = _wanted_files(repo, ignore)
        print(f"[fetch] {repo} -> {dest} ({len(files)} files)", flush=True)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_download_file, curl, repo, filename, dest, expected_size): filename
                for filename, expected_size in files
                if not _is_complete(dest / filename, expected_size)
            }
            for future in as_completed(futures):
                future.result()
        print(f"[done]  {dest} ({_dir_size_gb(dest)} GB)", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
