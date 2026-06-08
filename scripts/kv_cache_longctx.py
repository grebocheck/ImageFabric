#!/usr/bin/env python3
"""Hard long-context A/B for llama.cpp KV-cache quantization (needle-in-a-haystack).

This is the stress test the short-prompt A/B (`kv_cache_quality.py`) can't do:
it buries several "secret code" facts (needles) at known depths inside a long
filler passage (the haystack), then asks the model to recall each one. Long-range
attention over the cache is exactly where quantization hurts, so *retrieval
correctness vs depth* is an objective quality signal — not just divergence.

For each cache type it launches `llama-server`, runs every needle question
through the chat endpoint (jinja template + reasoning, so gpt-oss/harmony works),
and scores how many codes it recalled verbatim. Also reports the KV-cache size
(the whole reason to quantize) and decode throughput.

  python scripts/kv_cache_longctx.py \
      --model models/llm/gpt-oss-20b-MXFP4.gguf \
      --bin bin/llama-turbo/llama-server.exe \
      --types f16,q8_0,turbo3,turbo4 --ctx 16384 --haystack-tokens 6000 --needles 6

Self-contained (stdlib only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# Distinct, unlikely-to-be-guessed needle topics + code alphabet.
TOPICS = [
    "the Voss reactor", "the Halberd protocol", "the Meridian vault",
    "the Cobalt archive", "the Tamarind ledger", "the Quill outpost",
    "the Sable relay", "the Onyx mandate", "the Verdant cache",
    "the Crimson ferry",
]
FILLER_SENTENCES = [
    "The quarterly logistics review noted that throughput remained within nominal bounds.",
    "Maintenance crews rotated shifts to keep the auxiliary coolant loops stable.",
    "An unrelated memo discussed cafeteria scheduling and parking allocations.",
    "The weather station logged mild winds and no precipitation for the period.",
    "Inventory reconciliation closed without discrepancies across the main depots.",
    "A training seminar covered standard safety procedures for routine operations.",
    "The archival team continued digitizing older paper records into the index.",
    "Network telemetry showed steady latency with occasional negligible spikes.",
    "Procurement finalized a contract for replacement filters and gaskets.",
    "The visitor log recorded ordinary deliveries and scheduled contractor visits.",
]


def build_haystack(rng: random.Random, approx_tokens: int, n_needles: int) -> tuple[str, list[dict]]:
    """Return (haystack_text, needles). approx_tokens ~ 0.75 * word_count."""
    target_words = int(approx_tokens * 0.75)
    topics = rng.sample(TOPICS, n_needles)
    needles = [
        {"topic": t, "code": f"{rng.choice('XYZQKVW')}{rng.choice('JLNR')}-{rng.randint(1000, 9999)}"}
        for t in topics
    ]
    # Build filler lines until we hit the word budget, then splice needles at
    # evenly spread depths (so we can see if deep needles fail first).
    lines: list[str] = []
    words = 0
    n = 0
    while words < target_words:
        s = f"[line {n:04d}] {rng.choice(FILLER_SENTENCES)}"
        lines.append(s)
        words += len(s.split())
        n += 1
    for i, needle in enumerate(needles):
        depth = int((i + 1) / (n_needles + 1) * len(lines))
        lines[depth] = f"[line {depth:04d}] IMPORTANT: The secret code for {needle['topic']} is {needle['code']}. Remember it exactly."
    return "\n".join(lines), needles


def _drain(stream, sink: list[str]) -> None:
    for raw in stream:
        sink.append(raw.rstrip("\n"))


def _parse_kv_mib(log_lines: list[str]) -> float | None:
    total, found = 0.0, False
    for line in log_lines:
        m = re.search(r"KV (?:self size|buffer size)\s*=\s*([\d.]+)\s*MiB", line)
        if m:
            total += float(m.group(1))
            found = True
    return total if found else None


def _vram_oom(log_lines: list[str]) -> bool:
    return any(re.search(r"out of memory|failed to allocate|CUDA error", ln, re.I) for ln in log_lines)


def _wait_health(port: int, proc: subprocess.Popen, log: list[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "\n".join(log[-30:])
            extra = " (CUDA OOM — cache too large for this type)" if _vram_oom(log) else ""
            raise RuntimeError(f"llama-server exited (code {proc.returncode}){extra}:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become healthy in time")


def _ask(
    port: int,
    system: str,
    user: str,
    max_tokens: int,
    seed: int,
    timeout: float,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    min_p: float | None,
) -> dict:
    payload = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if top_p is not None:
        payload["top_p"] = top_p
    if min_p is not None:
        payload["min_p"] = min_p
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    msg = body["choices"][0]["message"]
    text = (msg.get("content") or "")
    reasoning = (msg.get("reasoning_content") or "")
    usage = body.get("usage") or {}
    return {"text": text, "reasoning": reasoning, "completion_tokens": usage.get("completion_tokens")}


def run_type(args, cache_type: str, haystack: str, needles: list[dict]) -> dict:
    bin_path = Path(args.bin).resolve()
    model = Path(args.model).resolve()
    cmd = [
        str(bin_path), "-m", str(model),
        "--host", "127.0.0.1", "--port", str(args.port),
        "-ngl", str(args.ngl), "-c", str(args.ctx), "--fit", "off",
    ]
    if cache_type != "f16":
        # A type may be "q4_0" (K=V) or "q8_0/q4_0" (K=q8_0, V=q4_0) for the
        # popular precise-K / cheap-V split.
        k, v = cache_type.split("/", 1) if "/" in cache_type else (cache_type, cache_type)
        cmd += ["--cache-type-k", k, "--cache-type-v", v, "--flash-attn", "on"]
    cmd += args.server_arg

    log: list[str] = []
    print(f"\n=== {cache_type} === launching {' '.join(cmd[7:])}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=str(bin_path.parent), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
    )
    if proc.stdout is not None:
        threading.Thread(target=_drain, args=(proc.stdout, log), daemon=True).start()
    if proc.stderr is not None:
        threading.Thread(target=_drain, args=(proc.stderr, log), daemon=True).start()
    system = "You are a precise assistant. Answer with only the requested code, nothing else."
    try:
        _wait_health(args.port, proc, log, args.load_timeout)
        if args.post_health_wait > 0:
            time.sleep(args.post_health_wait)
        kv_mib = _parse_kv_mib(log)
        correct = 0
        correct_any = 0
        speeds: list[float] = []
        details: list[dict] = []
        for i, needle in enumerate(needles):
            q = (
                f"{haystack}\n\n"
                f"Question: What is the secret code for {needle['topic']}? "
                f"Reply with only the code."
            )
            t0 = time.monotonic()
            res = _ask(
                args.port,
                system,
                q,
                args.max_tokens,
                args.seed,
                args.req_timeout,
                args.temperature,
                args.top_k,
                args.top_p,
                args.min_p,
            )
            dt = time.monotonic() - t0
            content_hit = needle["code"] in res["text"]
            reasoning_hit = needle["code"] in res["reasoning"]
            hit = content_hit or reasoning_hit
            correct += int(content_hit)
            correct_any += int(hit)
            if res["completion_tokens"] and dt > 0:
                speeds.append(res["completion_tokens"] / dt)
            depth_pct = int((i + 1) / (len(needles) + 1) * 100)
            details.append({"topic": needle["topic"], "code": needle["code"],
                            "depth_pct": depth_pct, "hit": hit, "content_hit": content_hit,
                            "reasoning_hit": reasoning_hit, "answer": res["text"].strip()[:80],
                            "reasoning": res["reasoning"].strip()[:160]})
            verdict = "OK " if content_hit else ("THINK" if reasoning_hit else "MISS")
            print(f"  needle {i + 1}/{len(needles)} (~{depth_pct}% deep): "
                  f"{verdict} -> {res['text'].strip()[:60]!r}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return {
        "cache_type": cache_type, "kv_mib": kv_mib,
        "correct": correct, "correct_any": correct_any, "total": len(needles),
        "mean_tok_per_s": (sum(speeds) / len(speeds)) if speeds else None,
        "details": details,
    }


def print_table(results: list[dict], baseline: str) -> None:
    base = next((r for r in results if r["cache_type"] == baseline and "error" not in r), None)
    base_kv = base["kv_mib"] if base else None
    print("\n" + "=" * 72)
    print(f"Long-context needle retrieval  (baseline KV = {baseline})")
    print("=" * 72)
    head = f"{'type':<9}{'KV MiB':>9}{'shrink':>8}{'tok/s':>8}{'final':>9}{'any':>8}"
    print(head + "\n" + "-" * len(head))
    for r in results:
        if "error" in r:
            print(f"{r['cache_type']:<9}{'FAILED':>9}   {r['error'][:38]}")
            continue
        kv = f"{r['kv_mib']:.0f}" if r.get("kv_mib") else "?"
        shrink = f"{base_kv / r['kv_mib']:.2f}x" if base_kv and r.get("kv_mib") else "?"
        tps = f"{r['mean_tok_per_s']:.1f}" if r.get("mean_tok_per_s") else "?"
        recall = f"{r['correct']}/{r['total']}"
        any_recall = f"{r.get('correct_any', r['correct'])}/{r['total']}"
        print(f"{r['cache_type']:<9}{kv:>9}{shrink:>8}{tps:>8}{recall:>9}{any_recall:>8}")
    print("-" * len(head))
    print("final = needles recalled in user-facing content. any = final or reasoning_content.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--types", default="f16,q8_0,turbo3,turbo4")
    p.add_argument("--baseline", default="f16")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--ngl", type=int, default=999)
    p.add_argument("--haystack-tokens", type=int, default=6000)
    p.add_argument("--needles", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--top-p", type=float)
    p.add_argument("--min-p", type=float)
    p.add_argument("--seed", type=int, default=20260608)
    p.add_argument("--port", type=int, default=8271)
    p.add_argument("--load-timeout", type=float, default=240.0)
    p.add_argument("--req-timeout", type=float, default=300.0)
    p.add_argument("--json-out")
    p.add_argument("--post-health-wait", type=float, default=1.0)
    p.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra llama-server argument; repeat for flags with values.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.model, args.bin):
        if not Path(path).exists():
            print(f"not found: {path}", file=sys.stderr)
            return 2
    rng = random.Random(args.seed)
    haystack, needles = build_haystack(rng, args.haystack_tokens, args.needles)
    print(f"haystack ~{args.haystack_tokens} tokens, {len(needles)} needles, ctx {args.ctx}")
    for nd in needles:
        print(f"  needle: code {nd['code']} for {nd['topic']}")

    results: list[dict] = []
    for ct in [t.strip() for t in args.types.split(",") if t.strip()]:
        try:
            results.append(run_type(args, ct, haystack, needles))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {ct} failed: {type(exc).__name__}: {exc}", flush=True)
            results.append({"cache_type": ct, "error": f"{type(exc).__name__}: {exc}"})

    print_table(results, args.baseline)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
