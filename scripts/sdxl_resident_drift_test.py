#!/usr/bin/env python3
"""Run repeated SDXL jobs without unloading the resident pipeline.

This catches the long-session drift that a swap-loop test can miss: the backend
keeps the same SDXL model resident, queues many same-shape generations, then
checks whether process RSS, device VRAM, or job duration keep climbing.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import urllib.error
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


def health(base_url: str) -> dict[str, Any]:
    return request(base_url, "GET", "/api/health")


def ensure_queue_idle(base_url: str) -> None:
    jobs = request(base_url, "GET", "/api/jobs?limit=1000")
    active = [j for j in jobs if j.get("status") in {"queued", "running"}]
    if active:
        sample = ", ".join(f"{j['id']}:{j['status']}" for j in active[:5])
        raise RuntimeError(
            f"Queue is not idle ({len(active)} active job(s): {sample}). "
            "Wait/cancel them first, or pass --allow-existing-jobs."
        )


def choose_sdxl(models: list[dict[str, Any]], model_id: str | None) -> dict[str, Any]:
    if model_id:
        for model in models:
            if model.get("id") == model_id:
                if model.get("family") != "sdxl" or model.get("job_type") != "image":
                    raise RuntimeError(f"Model {model_id!r} is not an SDXL image model")
                return model
        raise RuntimeError(f"Unknown model id: {model_id}")

    candidates = [
        m for m in models
        if m.get("family") == "sdxl" and m.get("job_type") == "image"
    ]
    if not candidates:
        raise RuntimeError("No SDXL image model returned by /api/models")
    return sorted(candidates, key=lambda m: str(m.get("name") or "").lower())[0]


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


def job_duration_seconds(job: dict[str, Any], fallback: float) -> float:
    started = job.get("started_at")
    finished = job.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return fallback
    try:
        return (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        return fallback


def run_image_job(base_url: str, model_id: str, params: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    job = create_job(base_url, model_id, params)
    print(f"queued {job['id']} -> {model_id}", flush=True)
    done = wait_job(base_url, job["id"], timeout)
    wall = time.monotonic() - started
    return done, job_duration_seconds(done, wall)


def rss_gb(snapshot: dict[str, Any]) -> float:
    return float(snapshot["mem"]["ram"]["process_rss_gb"])


def used_vram_gb(snapshot: dict[str, Any]) -> float | None:
    vram = snapshot["mem"].get("vram")
    if not vram:
        return None
    return float(vram["used_gb"])


def print_mem(prefix: str, snapshot: dict[str, Any], *, duration: float | None = None) -> None:
    vram = used_vram_gb(snapshot)
    vram_text = "n/a" if vram is None else f"{vram:.2f} GB"
    duration_text = "" if duration is None else f", duration={duration:.1f}s"
    print(
        f"{prefix}: rss={rss_gb(snapshot):.2f} GB, vram_used={vram_text}{duration_text}",
        flush=True,
    )


def assert_resident(snapshot: dict[str, Any], model_id: str) -> None:
    gpu = snapshot.get("gpu") or {}
    if gpu.get("model_id") != model_id:
        raise AssertionError(
            f"Expected resident model {model_id!r}, got {gpu.get('model_id')!r}"
        )


def assert_memory_drift(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    ram_growth_gb: float,
    vram_growth_gb: float,
) -> None:
    rss_delta = rss_gb(current) - rss_gb(baseline)
    if rss_delta > ram_growth_gb:
        raise AssertionError(
            f"Process RSS grew by {rss_delta:.2f} GB, over limit {ram_growth_gb:.2f} GB"
        )

    base_vram = used_vram_gb(baseline)
    cur_vram = used_vram_gb(current)
    if base_vram is None or cur_vram is None:
        return
    vram_delta = cur_vram - base_vram
    if vram_delta > vram_growth_gb:
        raise AssertionError(
            f"VRAM grew by {vram_delta:.2f} GB, over limit {vram_growth_gb:.2f} GB"
        )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def assert_duration_drift(durations: list[float], *, slowdown_ratio: float, duration_slack_seconds: float) -> None:
    if slowdown_ratio <= 0 or len(durations) < 4:
        return
    window = max(2, len(durations) // 4)
    first = mean(durations[:window])
    last = mean(durations[-window:])
    if first <= 0:
        return
    if last > first * slowdown_ratio and (last - first) > duration_slack_seconds:
        raise AssertionError(
            f"Generation duration drifted from {first:.1f}s to {last:.1f}s "
            f"(ratio {last / first:.2f}x)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8260")
    parser.add_argument("--model", help="Explicit SDXL model id. Defaults to the first SDXL model.")
    parser.add_argument("--jobs", type=int, default=8, help="Measured jobs after warmup.")
    parser.add_argument("--warmup-jobs", type=int, default=1)
    parser.add_argument("--job-timeout", type=float, default=900.0)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--ram-growth-gb", type=float, default=1.0)
    parser.add_argument("--vram-growth-gb", type=float, default=1.0)
    parser.add_argument("--slowdown-ratio", type=float, default=2.5)
    parser.add_argument("--duration-slack-seconds", type=float, default=8.0)
    parser.add_argument("--prompt", default="resident SDXL drift test, studio product photo, clean lighting")
    parser.add_argument("--negative", default="")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--allow-stub", action="store_true")
    parser.add_argument("--allow-existing-jobs", action="store_true")
    parser.add_argument("--no-free-gpu-first", action="store_true")
    parser.add_argument("--free-gpu-end", action="store_true")
    parser.add_argument("--json-out", help="Optional path to write the measurement summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = health(args.base_url)
    if start.get("stub_mode") and not args.allow_stub:
        print(
            "Backend is in STUB mode. Re-run with HFAB_STUB_MODE=false, "
            "or pass --allow-stub for a pipeline-only smoke test.",
            file=sys.stderr,
        )
        return 2
    if not args.allow_existing_jobs:
        ensure_queue_idle(args.base_url)

    model = choose_sdxl(request(args.base_url, "GET", "/api/models"), args.model)
    print(f"Using SDXL model: {model['id']} ({model.get('name')})")

    if not args.no_free_gpu_first:
        request(args.base_url, "POST", "/api/gpu/free")
        time.sleep(args.cooldown)

    params = {
        "prompt": args.prompt,
        "negative": args.negative,
        "steps": args.steps,
        "guidance": args.guidance,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "batch_size": 1,
        "validation": "sdxl-resident-drift",
    }

    for idx in range(1, max(1, args.warmup_jobs) + 1):
        print(f"warmup {idx}/{max(1, args.warmup_jobs)}")
        run_image_job(args.base_url, model["id"], {**params, "seed": args.seed - idx}, args.job_timeout)
        time.sleep(args.cooldown)

    baseline = health(args.base_url)
    assert_resident(baseline, model["id"])
    print_mem("resident baseline", baseline)

    measurements: list[dict[str, Any]] = []
    durations: list[float] = []
    for idx in range(1, args.jobs + 1):
        print(f"measured {idx}/{args.jobs}")
        done, duration = run_image_job(
            args.base_url,
            model["id"],
            {**params, "seed": args.seed + idx},
            args.job_timeout,
        )
        time.sleep(args.cooldown)
        snap = health(args.base_url)
        assert_resident(snap, model["id"])
        assert_memory_drift(
            baseline,
            snap,
            ram_growth_gb=args.ram_growth_gb,
            vram_growth_gb=args.vram_growth_gb,
        )
        durations.append(duration)
        image_ids = (done.get("result") or {}).get("image_ids") or []
        measurements.append({
            "index": idx,
            "job_id": done["id"],
            "image_ids": image_ids,
            "duration_seconds": duration,
            "process_rss_gb": rss_gb(snap),
            "vram_used_gb": used_vram_gb(snap),
        })
        print_mem(f"after measured {idx}", snap, duration=duration)

    assert_duration_drift(
        durations,
        slowdown_ratio=args.slowdown_ratio,
        duration_slack_seconds=args.duration_slack_seconds,
    )

    final = health(args.base_url)
    summary = {
        "model": model,
        "params": params,
        "baseline": {
            "process_rss_gb": rss_gb(baseline),
            "vram_used_gb": used_vram_gb(baseline),
        },
        "final": {
            "process_rss_gb": rss_gb(final),
            "vram_used_gb": used_vram_gb(final),
        },
        "rss_growth_gb": rss_gb(final) - rss_gb(baseline),
        "vram_growth_gb": None
        if used_vram_gb(baseline) is None or used_vram_gb(final) is None
        else used_vram_gb(final) - used_vram_gb(baseline),
        "durations_seconds": durations,
        "measurements": measurements,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.free_gpu_end:
        request(args.base_url, "POST", "/api/gpu/free")
    print("SDXL resident drift test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
