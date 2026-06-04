#!/usr/bin/env python3
"""Run the P0.3 LLM -> FLUX -> SDXL -> LLM swap-loop leak test.

The script talks to a running HFabric backend. It queues one job at a time
to force the exact swap order, does warmup cycles by default so lazy imports
and one-time backend caches are not counted as leaks, frees the GPU at the end of each measured cycle, then
asserts that process RSS and device VRAM return close to the warm baseline.
"""

from __future__ import annotations

import argparse
import json
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


def choose_model(models: list[dict[str, Any]], family: str, *, quant: str | None = None) -> dict[str, Any]:
    candidates = [m for m in models if m.get("family") == family]
    if quant is not None:
        exact = [
            m for m in candidates
            if m.get("quant") == quant
            or (quant == "nunchaku" and str(m.get("quant") or "").startswith("nunchaku"))
        ]
        if exact:
            return sorted(exact, key=lambda m: str(m["name"]).lower())[0]
        raise RuntimeError(
            f"No {family} model with quant={quant!r} was returned by /api/models. "
            "Pass an explicit model id to override."
        )
    if family == "flux":
        safe = [m for m in candidates if not m.get("slow")]
        candidates = safe or candidates
    if not candidates:
        q = f" with quant={quant!r}" if quant else ""
        raise RuntimeError(f"No {family}{q} model was returned by /api/models")
    return sorted(candidates, key=lambda m: (bool(m.get("slow")), str(m["name"]).lower()))[0]


def create_job(base_url: str, job_type: str, model_id: str, params: dict[str, Any]) -> dict[str, Any]:
    payload = [{"type": job_type, "model_id": model_id, "params": params}]
    jobs = request(base_url, "POST", "/api/jobs", payload)
    return jobs[0]


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


def rss_gb(snapshot: dict[str, Any]) -> float:
    return float(snapshot["mem"]["ram"]["process_rss_gb"])


def available_ram_gb(snapshot: dict[str, Any]) -> float:
    return float(snapshot["mem"]["ram"]["available_gb"])


def used_vram_gb(snapshot: dict[str, Any]) -> float | None:
    vram = snapshot["mem"].get("vram")
    if not vram:
        return None
    return float(vram["used_gb"])


def print_mem(prefix: str, snapshot: dict[str, Any]) -> None:
    vram = used_vram_gb(snapshot)
    vram_text = "n/a" if vram is None else f"{vram:.2f} GB"
    print(
        f"{prefix}: rss={rss_gb(snapshot):.2f} GB, "
        f"ram_available={available_ram_gb(snapshot):.2f} GB, vram_used={vram_text}",
        flush=True,
    )


def assert_returned_to_baseline(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    ram_slack_gb: float,
    vram_slack_gb: float,
) -> None:
    rss_delta = rss_gb(current) - rss_gb(baseline)
    if rss_delta > ram_slack_gb:
        raise AssertionError(
            f"Process RSS grew by {rss_delta:.2f} GB, over slack {ram_slack_gb:.2f} GB"
        )

    base_vram = used_vram_gb(baseline)
    cur_vram = used_vram_gb(current)
    if base_vram is None or cur_vram is None:
        return
    vram_delta = cur_vram - base_vram
    if vram_delta > vram_slack_gb:
        raise AssertionError(
            f"VRAM grew by {vram_delta:.2f} GB, over slack {vram_slack_gb:.2f} GB"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8260")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--job-timeout", type=float, default=900.0)
    parser.add_argument("--cooldown", type=float, default=8.0)
    parser.add_argument("--ram-slack-gb", type=float, default=1.0)
    parser.add_argument("--vram-slack-gb", type=float, default=0.75)
    parser.add_argument("--image-steps", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--llm-model")
    parser.add_argument("--flux-model")
    parser.add_argument("--sdxl-model")
    parser.add_argument(
        "--strict-cold-baseline",
        action="store_true",
        help="Skip the warmup cycle and compare against the pre-import process baseline.",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=2,
        help="Unmeasured cycles before taking the leak baseline; ignored with --strict-cold-baseline.",
    )
    parser.add_argument("--allow-stub", action="store_true")
    parser.add_argument("--allow-existing-jobs", action="store_true")
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

    models = request(args.base_url, "GET", "/api/models")
    llm = {"id": args.llm_model} if args.llm_model else choose_model(models, "gguf")
    flux = {"id": args.flux_model} if args.flux_model else choose_model(models, "flux", quant="nunchaku")
    sdxl = {"id": args.sdxl_model} if args.sdxl_model else choose_model(models, "sdxl")

    print("Using models:")
    print(f"  LLM : {llm['id']}")
    print(f"  FLUX: {flux['id']}")
    print(f"  SDXL: {sdxl['id']}")

    llm_params = {
        "prompt": "Return a concise five-word image prompt for a memory test.",
        "max_tokens": 32,
        "temperature": 0.1,
    }
    image_params = {
        "prompt": "memory leak test image, simple studio lighting",
        "negative": "",
        "steps": args.image_steps,
        "guidance": 3.5,
        "width": args.image_size,
        "height": args.image_size,
        "seed": 12345,
        "batch_size": 1,
    }
    sequence = [
        ("LLM", "llm", llm["id"], llm_params),
        ("FLUX", "image", flux["id"], image_params),
        ("SDXL", "image", sdxl["id"], image_params),
        ("LLM", "llm", llm["id"], llm_params),
    ]

    request(args.base_url, "POST", "/api/gpu/free")
    time.sleep(args.cooldown)
    if not args.strict_cold_baseline:
        for warmup in range(1, max(0, args.warmup_cycles) + 1):
            print(
                f"warmup cycle {warmup}/{args.warmup_cycles} "
                "(lazy imports/cache excluded from leak baseline)"
            )
            run_sequence(args.base_url, sequence, args.job_timeout)
            request(args.base_url, "POST", "/api/gpu/free")
            time.sleep(args.cooldown)

    baseline = health(args.base_url)
    print_mem("baseline", baseline)

    for cycle in range(1, args.cycles + 1):
        print(f"cycle {cycle}/{args.cycles}")
        run_sequence(args.base_url, sequence, args.job_timeout)
        request(args.base_url, "POST", "/api/gpu/free")
        time.sleep(args.cooldown)
        snap = health(args.base_url)
        print_mem(f"after cycle {cycle}", snap)
        assert_returned_to_baseline(
            baseline,
            snap,
            ram_slack_gb=args.ram_slack_gb,
            vram_slack_gb=args.vram_slack_gb,
        )

    print("swap-loop leak test passed")
    return 0


def run_sequence(base_url: str, sequence: list[tuple[str, str, str, dict[str, Any]]], timeout: float) -> None:
    for label, job_type, model_id, params in sequence:
        job = create_job(base_url, job_type, model_id, params)
        print(f"  {label}: queued {job['id']}", flush=True)
        wait_job(base_url, job["id"], timeout)


if __name__ == "__main__":
    raise SystemExit(main())
