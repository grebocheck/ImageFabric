"""The single GPU worker + phase-batching scheduler.

One worker owns the GPU, so GPU work is naturally serialized. The scheduler
picks the next job to *minimize model swaps*: within the highest-priority tier it
prefers a job that needs the model already resident, then one of the same type,
then the oldest. The effect on a mixed batch is the intended flow — drain all
LLM jobs, swap once, drain all image jobs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any

from sqlalchemy import select

from ..backends.base import ImageBackend, LLMBackend
from ..backends.registry import ModelRegistry
from ..db.models import Image, Job
from ..db.session import session_scope
from ..services.rag_service import search_documents as run_rag_search
from .arbiter import GpuArbiter
from .enums import EventType, JobStatus, JobType
from .events import Event, EventBus


def _coerce_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, n))


@dataclass
class JobSnapshot:
    id: str
    type: JobType
    model_id: str
    params: dict[str, Any]


class Worker:
    def __init__(self, bus: EventBus, arbiter: GpuArbiter, registry: ModelRegistry) -> None:
        self._bus = bus
        self._arbiter = arbiter
        self._registry = registry
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._running = False

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="hfabric-worker")

    async def stop(self) -> None:
        self._running = False
        self._wakeup.set()
        if self._task:
            await self._task
        await self._arbiter.free_all()

    def notify(self) -> None:
        """Wake the worker (call after enqueue/cancel)."""
        self._wakeup.set()

    # ---------------------------------------------------------------- loop
    async def _loop(self) -> None:
        # On restart, recover jobs that were RUNNING when we died.
        await self._requeue_orphans()
        while self._running:
            snap = await self._pick_next()
            if snap is None:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._run(snap)

    async def _requeue_orphans(self) -> None:
        async with session_scope() as s:
            rows = (await s.execute(
                select(Job).where(Job.status == JobStatus.RUNNING)
            )).scalars().all()
            for job in rows:
                job.status = JobStatus.QUEUED
                job.progress = 0.0

    # --------------------------------------------------- phase-batch select
    async def _pick_next(self) -> JobSnapshot | None:
        async with session_scope() as s:
            rows = (await s.execute(
                select(Job)
                .where(Job.status == JobStatus.QUEUED)
                .order_by(Job.priority.desc(), Job.created_at.asc())
            )).scalars().all()
            if not rows:
                return None

            top_priority = rows[0].priority
            tier = [j for j in rows if j.priority == top_priority]

            cur = self._arbiter.current
            chosen = None
            if cur is not None:
                chosen = next((j for j in tier if j.model_id == cur.descriptor.id), None)
                if chosen is None:
                    chosen = next(
                        (j for j in tier if j.type == cur.descriptor.job_type), None
                    )
            if chosen is None:
                chosen = tier[0]

            chosen.status = JobStatus.RUNNING
            chosen.started_at = datetime.now(timezone.utc)
            # enum columns come back as plain strings from SQLite -> normalize so
            # identity checks (`snap.type is JobType.IMAGE`) work downstream.
            return JobSnapshot(chosen.id, JobType(chosen.type), chosen.model_id, dict(chosen.params))

    # ----------------------------------------------------------- run a job
    async def _run(self, snap: JobSnapshot) -> None:
        await self._bus.publish(Event(EventType.JOB_STARTED, job_id=snap.id, job_type=snap.type.value))
        try:
            backend = self._registry.get_backend(snap.model_id)
            await self._arbiter.ensure(backend)

            last_emit = 0.0

            async def progress(frac: float, note: str | None) -> None:
                nonlocal last_emit
                now = asyncio.get_running_loop().time()
                if now - last_emit >= 0.1 or frac >= 1.0:
                    last_emit = now
                    await self._bus.publish(Event(
                        EventType.JOB_PROGRESS, job_id=snap.id, progress=frac, note=note
                    ))

            if snap.type is JobType.IMAGE:
                assert isinstance(backend, ImageBackend)
                records = await backend.generate(self._with_lora_paths(snap.params), progress)
                await self._finish_image(snap, records)
            else:
                assert isinstance(backend, LLMBackend)

                async def on_token(tok: str) -> None:
                    await self._bus.publish(Event(EventType.LLM_TOKEN, job_id=snap.id, token=tok))

                text = await backend.complete(snap.params, on_token)
                await self._finish_llm(snap, text)

        except Exception as exc:  # noqa: BLE001
            await self._fail(snap, repr(exc))

    def _with_lora_paths(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_loras = params.get("loras") or []
        if not isinstance(raw_loras, list) or not raw_loras:
            return params
        lora_paths: dict[str, str] = {}
        for item in raw_loras:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            lora = self._registry.get_lora(item["id"])
            lora_paths[lora.id] = str(lora.path)
        if not lora_paths:
            return params
        return {**params, "_lora_paths": lora_paths}

    async def _finish_image(self, snap: JobSnapshot, records: list[dict[str, Any]]) -> None:
        image_ids: list[str] = []
        chat_md: str | None = None
        async with session_scope() as s:
            for rec in records:
                img = Image(
                    job_id=snap.id,
                    path=rec["path"],
                    thumb_path=rec.get("thumb_path"),
                    seed=rec.get("seed"),
                    width=rec.get("width"),
                    height=rec.get("height"),
                    params=rec.get("params", {}),
                )
                s.add(img)
                await s.flush()
                image_ids.append(img.id)
            job = await s.get(Job, snap.id)
            if job:
                job.status = JobStatus.DONE
                job.progress = 1.0
                job.result = {"image_ids": image_ids}
                job.finished_at = datetime.now(timezone.utc)
            # /image chat bridge: render the result inline in the conversation
            if snap.params.get("assistant_message_id"):
                prompt = str(snap.params.get("prompt", "")).strip()
                chat_md = "\n\n".join(f"![{prompt}](/api/images/{iid}/file)" for iid in image_ids) \
                    or "(no image produced)"
                from ..services import chat_service  # noqa: PLC0415
                await chat_service.finalize_assistant_message(
                    s, snap.params["assistant_message_id"], chat_md
                )
        for iid, rec in zip(image_ids, records):
            await self._bus.publish(Event(
                EventType.IMAGE_READY, job_id=snap.id, image_id=iid,
                thumb=rec.get("thumb_path"), path=rec["path"],
            ))
        done = Event(EventType.JOB_DONE, job_id=snap.id, job_type=snap.type.value)
        if chat_md is not None:
            done = Event(EventType.JOB_DONE, job_id=snap.id, job_type=snap.type.value, text=chat_md)
        await self._bus.publish(done)

    async def _finish_llm(self, snap: JobSnapshot, text: str) -> None:
        tool_call = self._parse_image_tool_call(text, snap)
        if tool_call is None:
            tool_call = await self._build_document_tool_call(text, snap)
        child_job_id: str | None = None
        child_job_type: JobType | None = None
        child_text: str | None = None
        async with session_scope() as s:
            job = await s.get(Job, snap.id)
            if job:
                job.status = JobStatus.DONE
                job.progress = 1.0
                job.finished_at = datetime.now(timezone.utc)
            if tool_call:
                child_job = Job(
                    type=tool_call["job_type"],
                    model_id=tool_call["model_id"],
                    params=tool_call["params"],
                    priority=0,
                    status=JobStatus.QUEUED,
                )
                s.add(child_job)
                await s.flush()
                child_job_id = child_job.id
                child_job_type = JobType(tool_call["job_type"])
                child_text = tool_call["pending_text"]
                if job:
                    job.result = {"text": text, "tool_call": tool_call["public"], "child_job_id": child_job_id}
                await self._write_chat_reply(s, snap, child_text)
            else:
                if job:
                    job.result = {"text": text}
                await self._write_chat_reply(s, snap, text)
        if child_job_id:
            await self._bus.publish(Event(
                EventType.JOB_CREATED,
                job_id=child_job_id,
                job_type=(child_job_type or JobType.LLM).value,
            ))
            await self._bus.publish(Event(
                EventType.JOB_DONE,
                job_id=snap.id,
                job_type=snap.type.value,
                text=child_text,
                tool_child_job_id=child_job_id,
            ))
            self.notify()
            return
        await self._bus.publish(Event(
            EventType.JOB_DONE, job_id=snap.id, job_type=snap.type.value, text=text
        ))

    def _parse_image_tool_call(self, text: str, snap: JobSnapshot) -> dict[str, Any] | None:
        config = snap.params.get("image_tool")
        if not isinstance(config, dict) or not config.get("model_id"):
            return None
        obj = self._extract_json_object(text)
        if not isinstance(obj, dict):
            return None
        tool = obj.get("tool") or obj.get("name")
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
        if tool != "generate_image" or not isinstance(args, dict):
            return None
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return None
        image_params: dict[str, Any] = {
            "prompt": prompt,
            "assistant_message_id": config.get("assistant_message_id"),
            "conversation_id": config.get("conversation_id"),
            "source_llm_job_id": snap.id,
        }
        negative = str(args.get("negative") or "").strip()
        if negative:
            image_params["negative"] = negative
        image_params["steps"] = _coerce_int(args.get("steps"), 12, min_value=1, max_value=80)
        image_params["width"] = _coerce_int(args.get("width"), 768, min_value=256, max_value=2048)
        image_params["height"] = _coerce_int(args.get("height"), 768, min_value=256, max_value=2048)
        if args.get("seed") is not None:
            image_params["seed"] = _coerce_int(args.get("seed"), -1, min_value=-1, max_value=2_147_483_647)
        image_params["tool_call"] = "generate_image"
        public = {
            "tool": "generate_image",
            "prompt": prompt,
            "negative": negative,
            "steps": image_params["steps"],
            "width": image_params["width"],
            "height": image_params["height"],
        }
        return {
            "job_type": JobType.IMAGE,
            "model_id": str(config["model_id"]),
            "params": image_params,
            "public": public,
            "pending_text": f"*generating image...*\n\n`{prompt}`",
        }

    async def _build_document_tool_call(self, text: str, snap: JobSnapshot) -> dict[str, Any] | None:
        config = snap.params.get("document_tool")
        if not isinstance(config, dict):
            return None
        obj = self._extract_json_object(text)
        if not isinstance(obj, dict):
            return None
        tool = obj.get("tool") or obj.get("name")
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
        if tool != "search_documents" or not isinstance(args, dict):
            return None
        query = str(args.get("query") or "").strip()
        if not query:
            return None
        top_k = _coerce_int(args.get("top_k"), int(config.get("top_k") or 5), min_value=1, max_value=20)

        try:
            async with session_scope() as s:
                result = await run_rag_search(s, query=query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            context = f"Document search failed: {exc}"
            results: list[dict[str, Any]] = []
        else:
            context = str(result.get("context") or "").strip()
            results = list(result.get("results") or [])

        if not context:
            context = "No matching local documents were found."

        messages = list(snap.params.get("messages") or [])
        messages.append({"role": "assistant", "content": text.strip()})
        messages.append({
            "role": "user",
            "content": (
                "Tool result from search_documents:\n\n"
                f"{context}\n\n"
                "Now answer the user's latest question using this retrieved context when relevant. "
                "Cite bracketed source numbers like [1] when you use a retrieved source. "
                "If the context is insufficient, say what is missing."
            ),
        })
        child_params = self._child_llm_params(snap, messages)
        child_params["tool_result"] = {
            "tool": "search_documents",
            "query": query,
            "top_k": top_k,
            "results": results,
        }
        public = {"tool": "search_documents", "query": query, "top_k": top_k, "matches": len(results)}
        return {
            "job_type": JobType.LLM,
            "model_id": snap.model_id,
            "params": child_params,
            "public": public,
            "pending_text": f"*searching documents...*\n\n`{query}`",
        }

    @staticmethod
    def _child_llm_params(snap: JobSnapshot, messages: list[dict[str, str]]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "messages": messages,
            "assistant_message_id": snap.params.get("assistant_message_id"),
            "conversation_id": snap.params.get("conversation_id"),
            "source_llm_job_id": snap.id,
            "temperature": snap.params.get("temperature", 0.8),
            "max_tokens": snap.params.get("max_tokens", 512),
        }
        for key in ("top_p", "top_k", "min_p", "repeat_penalty", "seed", "stop"):
            if snap.params.get(key) is not None:
                params[key] = snap.params[key]
        return params

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()
        decoder = json.JSONDecoder()
        for i, ch in enumerate(cleaned):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    async def _fail(self, snap: JobSnapshot, error: str) -> None:
        async with session_scope() as s:
            job = await s.get(Job, snap.id)
            if job:
                job.status = JobStatus.ERROR
                job.error = error
                job.finished_at = datetime.now(timezone.utc)
            if snap.params.get("assistant_message_id"):
                await self._write_chat_reply(s, snap, error, error=True)
        await self._bus.publish(Event(EventType.JOB_ERROR, job_id=snap.id, error=error))

    @staticmethod
    async def _write_chat_reply(s, snap: JobSnapshot, text: str, *, error: bool = False) -> None:
        """If this LLM job backs a chat message, persist the reply into it."""
        message_id = snap.params.get("assistant_message_id")
        if not message_id:
            return
        from ..services import chat_service  # noqa: PLC0415

        await chat_service.finalize_assistant_message(s, message_id, text, error=error)
