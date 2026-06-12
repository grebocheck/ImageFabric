"""Pidfile helpers for managed external server processes."""

from __future__ import annotations

import logging
from pathlib import Path
import time

import psutil

from ..config import settings

LLAMA_SERVER_PID = "llama-server.pid"


def llama_server_pidfile() -> Path:
    return settings.runtime_dir / LLAMA_SERVER_PID


def write_pidfile(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def remove_pidfile(path: Path) -> None:
    path.unlink(missing_ok=True)


def reap_pidfile(path: Path, expected_name: str, logger: logging.Logger) -> bool:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    try:
        pid = int(raw)
    except ValueError:
        remove_pidfile(path)
        return False

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        remove_pidfile(path)
        return False

    try:
        name = proc.name()
    except (psutil.Error, OSError):
        return False
    if expected_name.casefold() not in name.casefold():
        return False

    logger.warning(
        "event=process.reap pid=%s name=%s pidfile=%s expected=%s",
        pid,
        name,
        path,
        expected_name,
    )
    _terminate(proc)
    remove_pidfile(path)
    return True


def reap_known_pidfiles(logger: logging.Logger) -> None:
    reap_pidfile(llama_server_pidfile(), "llama-server", logger)


def _terminate(proc: psutil.Process, timeout: float = 5.0) -> None:
    try:
        children = proc.children(recursive=True)
    except psutil.Error:
        children = []
    targets = [*children, proc]
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(targets, timeout=timeout)
    if alive:
        for target in alive:
            try:
                target.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=timeout)
    # Give Windows a short moment to release process handles before the caller
    # starts a replacement server on the same port.
    if not gone:
        time.sleep(0.1)
