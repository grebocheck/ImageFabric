#!/usr/bin/env python3
"""Validate that a mixed batch needs exactly one LLM <-> image swap.

The check runs against a live ImageFabric backend. It subscribes to WebSocket
events, queues LLM/image/LLM/image in one REST request, and asserts that the
worker phase-batches them as LLM/LLM/image/image with only one model-family
transition.
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


def choose_model(models: list[dict[str, Any]], job_type: str) -> dict[str, Any]:
    candidates = [m for m in models if m.get("job_type") == job_type]
    if job_type == "image":
        preferred = [
            m for m in candidates
            if m.get("family") == "flux" and str(m.get("quant") or "").startswith("nunchaku")
        ]
        candidates = preferred or [m for m in candidates if not m.get("slow")] or candidates
    if not candidates:
        raise RuntimeError(f"No {job_type} model was returned by /api/models")
    return sorted(candidates, key=lambda m: (bool(m.get("slow")), str(m["name"]).lower()))[0]


def ensure_queue_idle(base_url: str) -> None:
    jobs = request(base_url, "GET", "/api/jobs?limit=1000")
    active = [j for j in jobs if j.get("status") in {"queued", "running"}]
    if active:
        sample = ", ".join(f"{j['id']}:{j['status']}" for j in active[:5])
        raise RuntimeError(
            f"Queue is not idle ({len(active)} active job(s): {sample}). "
            "Wait/cancel them first, or pass --allow-existing-jobs."
        )


def count_transitions(families: list[str]) -> int:
    compact: list[str] = []
    for family in families:
        if not compact or compact[-1] != family:
            compact.append(family)
    return max(0, len(compact) - 1)


async def run_check(args: argparse.Namespace) -> int:
    try:
        import websockets
    except ImportError:
        print(
            "Python package 'websockets' is required. It is installed by "
            "backend/requirements.txt via uvicorn[standard].",
            file=sys.stderr,
        )
        return 2

    if not args.allow_existing_jobs:
        await asyncio.to_thread(ensure_queue_idle, args.base_url)

    models = await asyncio.to_thread(request, args.base_url, "GET", "/api/models")
    llm = {"id": args.llm_model} if args.llm_model else choose_model(models, "llm")
    image = {"id": args.image_model} if args.image_model else choose_model(models, "image")

    await asyncio.to_thread(request, args.base_url, "POST", "/api/gpu/free")
    print(f"Using LLM   : {llm['id']}")
    print(f"Using image : {image['id']}")

    llm_params = {
        "prompt": "Return a terse image prompt for phase batching.",
        "max_tokens": 24,
        "temperature": 0.1,
    }
    image_params = {
        "prompt": "phase batching validation image",
        "negative": "",
        "steps": args.image_steps,
        "guidance": 3.5,
        "width": args.image_size,
        "height": args.image_size,
        "seed": 4242,
        "batch_size": 1,
    }
    payload = [
        {"type": "llm", "model_id": llm["id"], "params": llm_params},
        {"type": "image", "model_id": image["id"], "params": image_params},
        {"type": "llm", "model_id": llm["id"], "params": llm_params},
        {"type": "image", "model_id": image["id"], "params": image_params},
    ]

    started_types: list[str] = []
    loaded_families: list[str] = []
    done: set[str] = set()
    errors: list[str] = []

    async with websockets.connect(websocket_url(args.base_url)) as ws:
        jobs = await asyncio.to_thread(request, args.base_url, "POST", "/api/jobs", payload)
        expected_ids = {j["id"] for j in jobs}
        print("Queued jobs:")
        for job in jobs:
            print(f"  {job['id']} {job['type']}")

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and len(done) < len(expected_ids) and not errors:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, deadline - time.monotonic()))
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "job.started" and event.get("job_id") in expected_ids:
                started_types.append(str(event.get("job_type")))
            elif etype == "model.loaded":
                family = event.get("family")
                if family:
                    loaded_families.append(str(family))
            elif etype == "job.done" and event.get("job_id") in expected_ids:
                done.add(str(event["job_id"]))
            elif etype == "job.error" and event.get("job_id") in expected_ids:
                errors.append(f"{event.get('job_id')}: {event.get('error')}")

    if errors:
        raise RuntimeError("Batch failed: " + "; ".join(errors))
    if len(done) != 4:
        raise TimeoutError(f"Only {len(done)}/4 jobs completed within {args.timeout:.0f}s")

    transitions = count_transitions(loaded_families)
    print(f"Started types : {started_types}")
    print(f"Loaded families: {loaded_families}")
    print(f"Family swaps  : {transitions}")

    if started_types != ["llm", "llm", "image", "image"]:
        raise AssertionError("Expected start order llm,llm,image,image")
    if transitions != 1:
        raise AssertionError(f"Expected exactly one LLM <-> image swap, got {transitions}")

    print("phase-batching check passed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8260")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--image-steps", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--llm-model")
    parser.add_argument("--image-model")
    parser.add_argument("--allow-existing-jobs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_check(args))


if __name__ == "__main__":
    raise SystemExit(main())
