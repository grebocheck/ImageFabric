from __future__ import annotations

import logging
import os

import psutil

from app.util.pidfiles import reap_pidfile, write_pidfile


def test_dead_pidfile_is_cleaned(tmp_path):
    path = tmp_path / "llama-server.pid"
    dead_pid = max(psutil.pids()) + 100_000
    write_pidfile(path, dead_pid)

    assert reap_pidfile(path, "llama-server", logging.getLogger("test")) is False
    assert not path.exists()


def test_alive_wrong_name_pidfile_is_left_alone(tmp_path):
    path = tmp_path / "llama-server.pid"
    write_pidfile(path, os.getpid())

    assert reap_pidfile(path, "llama-server", logging.getLogger("test")) is False
    assert path.exists()
