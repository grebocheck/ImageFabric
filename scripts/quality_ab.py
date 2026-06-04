#!/usr/bin/env python3
"""Queue a same-seed image quality A/B run across local image models.

The script talks to a running HFabric backend. It is intentionally simple:
same prompt, seed, size, steps, and guidance across each selected image model,
then it prints job/image ids and gallery URLs for side-by-side review.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
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


def ensure_queue_idle(base_url: str) -> None:
    jobs = request(base_url, "GET", "/api/jobs?limit=1000")
    active = [j for j in jobs if j.get("status") in {"queued", "running"}]
    if active:
        sample = ", ".join(f"{j['id']}:{j['status']}" for j in active[:5])
        raise RuntimeError(
            f"Queue is not idle ({len(active)} active job(s): {sample}). "
            "Wait/cancel them first, or pass --allow-existing-jobs."
        )


def image_models(models: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        found = [m for m in models if m.get("id") in wanted]
        missing = sorted(wanted - {str(m.get("id")) for m in found})
        if missing:
            raise RuntimeError(f"Unknown image model id(s): {', '.join(missing)}")
        return found

    candidates = [m for m in models if m.get("job_type") == "image"]
    if args.family != "all":
        candidates = [m for m in candidates if m.get("family") == args.family]
    if not args.include_slow:
        safe = [m for m in candidates if not m.get("slow")]
        candidates = safe or candidates
    candidates = sorted(
        candidates,
        key=lambda m: (
            str(m.get("family") or ""),
            str(m.get("quant") or ""),
            bool(m.get("slow")),
            str(m.get("name") or "").lower(),
        ),
    )
    if args.limit:
        candidates = candidates[:args.limit]
    return candidates


def create_job(base_url: str, model_id: str, params: dict[str, Any]) -> dict[str, Any]:
    payload = [{"type": "image", "model_id": model_id, "params": params}]
    return request(base_url, "POST", "/api/jobs", payload)[0]


def wait_job(base_url: str, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request(base_url, "GET", f"/api/jobs/{job_id}")
        status = job.get("status")
        if status == "done":
            return job
        if status in {"error", "cancelled"}:
            raise RuntimeError(f"Job {job_id} ended as {status}: {job.get('error')}")
        time.sleep(1.0)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout:.0f}s")


def job_duration_seconds(job: dict[str, Any]) -> float | None:
    started = job.get("started_at")
    finished = job.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        return (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8260")
    parser.add_argument("--family", default="flux", choices=["flux", "sdxl", "all"])
    parser.add_argument("--models", help="Comma-separated image model ids. Overrides --family/--limit.")
    parser.add_argument("--limit", type=int, default=0, help="Auto-selected model limit; 0 means all candidates.")
    parser.add_argument("--prompt", default="quality comparison portrait, detailed fabric, cinematic studio lighting")
    parser.add_argument("--negative", default="")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--job-timeout", type=float, default=900.0)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--allow-stub", action="store_true")
    parser.add_argument("--allow-existing-jobs", action="store_true")
    parser.add_argument("--free-gpu-first", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--json-out", help="Optional path to write the result summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    health = request(args.base_url, "GET", "/api/health")
    if health.get("stub_mode") and not args.allow_stub:
        print(
            "Backend is in STUB mode. Re-run with HFAB_STUB_MODE=false, "
            "or pass --allow-stub for a pipeline-only smoke test.",
            file=sys.stderr,
        )
        return 2
    if not args.allow_existing_jobs:
        ensure_queue_idle(args.base_url)
    if args.free_gpu_first:
        request(args.base_url, "POST", "/api/gpu/free")

    models = request(args.base_url, "GET", "/api/models")
    candidates = image_models(models, args)
    if len(candidates) < 2 and not args.models:
        raise RuntimeError("Need at least two image models for an automatic A/B run; pass --models to override.")

    params = {
        "prompt": args.prompt,
        "negative": args.negative,
        "steps": args.steps,
        "guidance": args.guidance,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "batch_size": 1,
        "ab_group": f"ab-{int(time.time())}",
    }
    summary: list[dict[str, Any]] = []
    print("A/B candidates:")
    for model in candidates:
        print(f"  {model['id']} ({model.get('family')}, {model.get('quant') or 'base'})")

    for model in candidates:
        started = time.monotonic()
        job = create_job(args.base_url, model["id"], params)
        print(f"queued {job['id']} -> {model['id']}", flush=True)
        try:
            done = wait_job(args.base_url, job["id"], args.job_timeout)
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            elapsed = time.monotonic() - started
            summary.append({
                "model_id": model["id"],
                "model": model.get("name"),
                "family": model.get("family"),
                "quant": model.get("quant"),
                "job_id": job["id"],
                "image_ids": [],
                "urls": [],
                "elapsed_seconds": elapsed,
                "job_duration_seconds": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"error {job['id']} -> {model['id']}: {exc}", flush=True)
            continue
        elapsed = time.monotonic() - started
        image_ids = (done.get("result") or {}).get("image_ids") or []
        summary.append({
            "model_id": model["id"],
            "model": model.get("name"),
            "family": model.get("family"),
            "quant": model.get("quant"),
            "job_id": job["id"],
            "image_ids": image_ids,
            "urls": [f"{args.base_url.rstrip('/')}/api/images/{iid}/file" for iid in image_ids],
            "elapsed_seconds": elapsed,
            "job_duration_seconds": job_duration_seconds(done),
        })

    print(json.dumps(summary, indent=2))
    if args.json_out:
        from pathlib import Path

        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
