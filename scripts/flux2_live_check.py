#!/usr/bin/env python3
"""Run a live FLUX.2 [klein] validation against a running backend.

The check queues one image job through the normal HFabric REST API, listens
to WebSocket events, samples /api/health for RAM/VRAM peaks, and writes a compact
JSON report. It validates the real queue -> arbiter -> diffusers path instead of
calling the backend class directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def request(base_url: str, method: str, path: str, payload: Any | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code} {detail}") from exc
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def websocket_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    return f"{scheme}://{netloc}/ws"


def ensure_queue_idle(base_url: str) -> None:
    jobs = request(base_url, "GET", "/api/jobs?limit=1000")
    active = [j for j in jobs if j.get("status") in {"queued", "running"}]
    if active:
        sample = ", ".join(f"{j['id']}:{j['status']}" for j in active[:5])
        raise RuntimeError(f"Queue is not idle ({len(active)} active job(s): {sample})")


def choose_flux2(models: list[dict[str, Any]], model_id: str | None) -> dict[str, Any]:
    if model_id:
        for model in models:
            if model.get("id") == model_id:
                if model.get("family") != "flux2":
                    raise RuntimeError(f"Model {model_id!r} is {model.get('family')}, not flux2")
                return model
        raise RuntimeError(f"Unknown model id: {model_id}")
    candidates = [m for m in models if m.get("job_type") == "image" and m.get("family") == "flux2"]
    if not candidates:
        raise RuntimeError("No FLUX.2 image model returned by /api/models")
    return sorted(
        candidates,
        key=lambda m: (
            0 if str(m.get("quant", "")).startswith("nunchaku") else 1,
            str(m.get("name", "")).lower(),
        ),
    )[0]


def peak(samples: list[dict[str, Any]], path: tuple[str, ...], default: float = 0.0) -> float:
    values: list[float] = []
    for sample in samples:
        value: Any = sample
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return max(values) if values else default


def valley(samples: list[dict[str, Any]], path: tuple[str, ...], default: float = 0.0) -> float:
    values: list[float] = []
    for sample in samples:
        value: Any = sample
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return min(values) if values else default


async def sample_health(base_url: str, stop: asyncio.Event, interval: float) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while not stop.is_set():
        try:
            samples.append(await asyncio.to_thread(request, base_url, "GET", "/api/health", None, 10.0))
        except Exception as exc:  # noqa: BLE001
            samples.append({"error": repr(exc)})
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    return samples


async def run_check(args: argparse.Namespace) -> int:
    try:
        import websockets
    except ImportError:
        print("Python package 'websockets' is required.", file=sys.stderr)
        return 2

    health = request(args.base_url, "GET", "/api/health")
    if health.get("stub_mode") and not args.allow_stub:
        print("Backend is in STUB mode. Start with HFAB_STUB_MODE=false.", file=sys.stderr)
        return 2
    if not args.allow_existing_jobs:
        ensure_queue_idle(args.base_url)
    if args.free_gpu_first:
        request(args.base_url, "POST", "/api/gpu/free")

    model = choose_flux2(request(args.base_url, "GET", "/api/models"), args.model)
    params = {
        "prompt": args.prompt,
        "negative": args.negative,
        "steps": args.steps,
        "guidance": args.guidance,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "batch_size": 1,
        "validation": "p3-flux2-live-check",
    }
    payload = [{"type": "image", "model_id": model["id"], "params": params}]

    events: list[dict[str, Any]] = []
    progress_notes: list[str] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_health(args.base_url, stop, args.sample_interval))
    started_at = time.monotonic()
    job_id = ""
    image_ids: list[str] = []

    try:
        async with websockets.connect(websocket_url(args.base_url), ping_interval=20, ping_timeout=20) as ws:
            job = (await asyncio.to_thread(request, args.base_url, "POST", "/api/jobs", payload))[0]
            job_id = job["id"]
            print(f"queued {job_id} -> {model['id']}", flush=True)
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, deadline - time.monotonic()))
                event = json.loads(raw)
                etype = event.get("type")
                if etype in {"model.loading", "model.loaded", "job.started", "job.progress", "image.ready", "job.done", "job.error"}:
                    events.append(event)
                if etype == "job.progress" and event.get("job_id") == job_id:
                    note = event.get("note")
                    if isinstance(note, str):
                        progress_notes.append(note)
                if etype == "image.ready" and event.get("job_id") == job_id:
                    image_ids.append(str(event.get("image_id")))
                if etype == "job.done" and event.get("job_id") == job_id:
                    break
                if etype == "job.error" and event.get("job_id") == job_id:
                    raise RuntimeError(str(event.get("error")))
            else:
                raise TimeoutError(f"Job {job_id} did not finish within {args.timeout:.0f}s")
    finally:
        stop.set()

    elapsed = time.monotonic() - started_at
    samples = await sampler
    final_job = request(args.base_url, "GET", f"/api/jobs/{job_id}")
    if final_job.get("status") != "done":
        raise RuntimeError(f"Job {job_id} ended as {final_job.get('status')}: {final_job.get('error')}")
    image_ids = image_ids or (final_job.get("result") or {}).get("image_ids") or []
    load_event = next((e for e in events if e.get("type") == "model.loaded" and e.get("family") == "flux2"), {})

    summary = {
        "model": model,
        "job_id": job_id,
        "image_ids": image_ids,
        "image_urls": [f"{args.base_url.rstrip('/')}/api/images/{iid}/file" for iid in image_ids],
        "params": params,
        "elapsed_seconds": elapsed,
        "peak_process_rss_gb": peak(samples, ("mem", "ram", "process_rss_gb")),
        "peak_ram_used_gb": peak(samples, ("mem", "ram", "used_gb")),
        "min_ram_available_gb": valley(samples, ("mem", "ram", "available_gb")),
        "peak_vram_used_gb": peak(samples, ("mem", "vram", "used_gb")),
        "min_vram_free_gb": valley(samples, ("mem", "vram", "free_gb")),
        "load_report": load_event.get("load_report"),
        "progress_notes": progress_notes[-8:],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8260")
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="A compact product photo of a translucent glass teapot on a steel table, soft window light")
    parser.add_argument("--negative", default="")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--json-out")
    parser.add_argument("--allow-stub", action="store_true")
    parser.add_argument("--allow-existing-jobs", action="store_true")
    parser.add_argument("--free-gpu-first", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_check(parse_args())))
