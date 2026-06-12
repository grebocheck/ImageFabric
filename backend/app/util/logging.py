"""Rotating file logging and event-bus log subscriber."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import Any

from ..config import Settings
from ..core.enums import EventType
from ..core.events import Event, EventBus

LOG_NAME = "hfabric"
LOG_FILE_NAME = "hfabric.log"
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5


def configure_file_logging(settings: Settings) -> logging.Logger:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / LOG_FILE_NAME
    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = True

    for handler in list(logger.handlers):
        if getattr(handler, "_hfabric_file_handler", False):
            if getattr(handler, "baseFilename", None) == str(log_path):
                return logger
            logger.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        log_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler._hfabric_file_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def install_unhandled_exception_logging(logger: logging.Logger, loop: asyncio.AbstractEventLoop) -> None:
    def excepthook(exc_type, exc, tb) -> None:
        logger.critical("event=unhandled.exception", exc_info=(exc_type, exc, tb))

    def loop_handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        message = context.get("message") or "unhandled event loop exception"
        if exc is not None:
            logger.error(
                "event=unhandled.async message=%s",
                message,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            logger.error("event=unhandled.async message=%s context=%s", message, _json(context))

    sys.excepthook = excepthook
    loop.set_exception_handler(loop_handler)


class EventLogSubscriber:
    def __init__(self, bus: EventBus, logger: logging.Logger, settings: Settings) -> None:
        self._bus = bus
        self._logger = logger
        self._settings = settings
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._job_started: dict[str, float] = {}

    async def start(self) -> None:
        self._log_startup()
        self._task = asyncio.create_task(self._run(), name="hfabric-event-log")
        await self._ready.wait()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    def _log_startup(self) -> None:
        s = self._settings
        payload = {
            "host": s.host,
            "port": s.port,
            "stub_mode": s.stub_mode,
            "data_dir": str(s.data_dir),
            "outputs_dir": str(s.outputs_dir),
            "logs_dir": str(s.logs_dir),
            "runtime_dir": str(s.runtime_dir),
            "image_models_dir": str(s.image_models_dir),
            "lora_models_dir": str(s.lora_models_dir),
            "llm_models_dir": str(s.llm_models_dir),
            "tts_models_dir": str(s.tts_models_dir),
            "transcription_models_dir": str(s.transcription_models_dir),
            "embed_models_dir": str(s.embed_models_dir),
            "vision_models_dir": str(s.vision_models_dir),
            "voice_models_dir": str(s.voice_models_dir),
        }
        self._logger.info("event=startup.config %s", _json(payload))

    async def _run(self) -> None:
        async with self._bus.subscribe() as q:
            self._ready.set()
            while True:
                event = await q.get()
                self._handle(event)

    def _handle(self, event: Event) -> None:
        event_type = event.get("type")
        if event_type == EventType.ARBITER_NOTE.value:
            self._logger.info("event=arbiter.note %s", _json_payload(event))
            return
        if event_type == EventType.JOB_STARTED.value:
            job_id = str(event.get("job_id") or "")
            if job_id:
                self._job_started[job_id] = float(event.get("ts") or 0.0)
            self._logger.info("event=job.started %s", _json_payload(event))
            return
        if event_type in {
            EventType.JOB_DONE.value,
            EventType.JOB_ERROR.value,
            EventType.JOB_CANCELLED.value,
        }:
            payload = dict(_payload(event))
            job_id = str(event.get("job_id") or "")
            started = self._job_started.pop(job_id, None)
            if started is not None:
                payload["duration_s"] = round(float(event.get("ts") or 0.0) - started, 3)
            level = logging.ERROR if event_type == EventType.JOB_ERROR.value else logging.INFO
            self._logger.log(level, "event=%s %s", event_type, _json(payload))
            return
        if event_type in {EventType.MODEL_LOADED.value, EventType.MODEL_UNLOADED.value}:
            self._logger.info("event=%s %s", event_type, _json_payload(event))
            return
        if event_type in {
            EventType.VOICE_SESSION_STARTED.value,
            EventType.VOICE_SESSION_STOPPED.value,
        }:
            self._logger.info("event=%s %s", event_type, _json_payload(event))


def _payload(event: Event) -> dict[str, Any]:
    return {k: v for k, v in event.items() if k not in {"type", "ts"}}


def _json_payload(event: Event) -> str:
    payload = _payload(event)
    if "ts" in event:
        payload["ts"] = event["ts"]
    return _json(payload)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
