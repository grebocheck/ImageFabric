#!/usr/bin/env python3
"""Quality / cost A/B for llama.cpp KV-cache ("context") quantization types.

Holds the model and binary fixed and varies ONLY ``--cache-type-k/-v`` so the
single variable under test is the cache quantization. For each type it launches
``llama-server``, greedily (temperature 0, fixed seed) completes a battery of
prompts, and records:

  * KV-cache size in MiB (parsed from the server's startup log) — the *benefit*
  * decode throughput (tokens/s)                                 — the *cost*
  * how far the output drifts from the f16 full-precision run    — the *quality*

f16 is the gold reference: a quantized cache that reproduces f16's greedy output
verbatim has lost nothing; the more it diverges, the more the cache hurt. Run it
against a TurboQuant-patched build to score turbo2/turbo3/turbo4:

  python scripts/kv_cache_quality.py \
      --model models/llm/gemma-3-12b-it-heretic-v2-Q4_K_M.gguf \
      --bin bin/llama-turbo/llama-server.exe \
      --types f16,q8_0,turbo3,turbo4

Self-contained (stdlib only); does NOT go through the HFabric backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# Prompts chosen to exercise the cache: a couple of long-context passages (where
# attention over a quantized cache matters most) plus short factual/coding ones.
DEFAULT_PROMPTS: list[str] = [
    "Explain in three sentences why the sky appears blue during the day and red at sunset.",
    "Write a Python function `is_palindrome(s)` that ignores case and non-alphanumeric characters.",
    "List the first 10 prime numbers, then state their sum.",
    (
        "Read the passage and answer the question.\n\n"
        "Passage: The Antikythera mechanism is an ancient Greek hand-powered "
        "orrery, described as the oldest known analogue computer. It was used to "
        "predict astronomical positions and eclipses decades in advance, and to "
        "track the four-year cycle of athletic games similar to an Olympiad. It "
        "was recovered in 1901 from the Antikythera shipwreck off the Greek "
        "island of Antikythera. The instrument is believed to have been designed "
        "and constructed by Greek scientists and has been variously dated to "
        "about 87 BC, or between 150 and 100 BC.\n\n"
        "Question: What was the Antikythera mechanism used to predict, and when "
        "was it recovered?"
    ),
    (
        "Summarize the following text in one sentence.\n\n"
        "Photosynthesis is a process used by plants and other organisms to "
        "convert light energy into chemical energy that, through cellular "
        "respiration, can later be released to fuel the organism's activities. "
        "This chemical energy is stored in carbohydrate molecules, such as "
        "sugars and starches, which are synthesized from carbon dioxide and "
        "water."
    ),
]


def _drain(stream, sink: list[str]) -> None:
    for raw in stream:
        sink.append(raw.rstrip("\n"))


def _parse_kv_mib(log_lines: list[str]) -> float | None:
    """Pull the total KV-cache size (MiB) out of llama.cpp's startup log."""
    total = 0.0
    found = False
    for line in log_lines:
        # e.g. "llama_kv_cache: CUDA0 KV buffer size =   512.00 MiB"
        #      "llama_kv_cache_unified: KV self size  =  256.00 MiB, K (f16): ..."
        m = re.search(r"KV (?:self size|buffer size)\s*=\s*([\d.]+)\s*MiB", line)
        if m:
            total += float(m.group(1))
            found = True
    return total if found else None


def _wait_health(port: int, proc: subprocess.Popen, log: list[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "\n".join(log[-30:])
            raise RuntimeError(f"llama-server exited (code {proc.returncode}) before healthy:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become healthy in time")


def _complete(port: int, prompt: str, n_predict: int, seed: int, timeout: float) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,   # greedy / deterministic
        "top_k": 1,
        "seed": seed,
        "cache_prompt": False,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    timings = body.get("timings") or {}
    return {
        "content": body.get("content", ""),
        "tok_per_s": timings.get("predicted_per_second"),
        "predicted_n": timings.get("predicted_n"),
    }


def _char_prefix_ratio(a: str, b: str) -> float:
    """Fraction of the reference string reproduced before the first divergence."""
    if not a:
        return 1.0 if not b else 0.0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / len(a)


def run_type(args, cache_type: str) -> dict:
    bin_path = Path(args.bin).resolve()
    model = Path(args.model).resolve()
    cmd = [
        str(bin_path),
        "-m", str(model),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
        "--fit", "off",
    ]
    if cache_type != "f16":
        cmd += ["--cache-type-k", cache_type, "--cache-type-v", cache_type, "--flash-attn", "on"]

    log: list[str] = []
    print(f"\n=== {cache_type} === launching {' '.join(cmd[3:])}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(bin_path.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    drain = threading.Thread(target=_drain, args=(proc.stderr, log), daemon=True)
    drain.start()
    try:
        _wait_health(args.port, proc, log, args.load_timeout)
        kv_mib = _parse_kv_mib(log)
        outputs: list[dict] = []
        speeds: list[float] = []
        t0 = time.monotonic()
        for i, prompt in enumerate(args.prompts):
            res = _complete(args.port, prompt, args.n_predict, args.seed, args.req_timeout)
            outputs.append({"prompt_index": i, "content": res["content"]})
            if res["tok_per_s"]:
                speeds.append(res["tok_per_s"])
            print(f"  [{i + 1}/{len(args.prompts)}] {res['predicted_n']} tok "
                  f"@ {res['tok_per_s']:.1f} tok/s" if res["tok_per_s"] else f"  [{i + 1}] done",
                  flush=True)
        wall = time.monotonic() - t0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return {
        "cache_type": cache_type,
        "kv_mib": kv_mib,
        "mean_tok_per_s": (sum(speeds) / len(speeds)) if speeds else None,
        "wall_seconds": round(wall, 1),
        "outputs": outputs,
    }


def score_against_baseline(results: list[dict], baseline_type: str) -> None:
    base = next((r for r in results if r["cache_type"] == baseline_type), None)
    if base is None:
        print(f"\n(no baseline '{baseline_type}' run — skipping quality scoring)")
        return
    base_out = {o["prompt_index"]: o["content"] for o in base["outputs"]}
    base_kv = base["kv_mib"]
    for r in results:
        exact = 0
        ratios: list[float] = []
        for o in r["outputs"]:
            ref = base_out.get(o["prompt_index"], "")
            if o["content"] == ref:
                exact += 1
            ratios.append(_char_prefix_ratio(ref, o["content"]))
        r["exact_match"] = exact
        r["n_prompts"] = len(r["outputs"])
        r["mean_prefix_agreement"] = sum(ratios) / len(ratios) if ratios else None
        if base_kv and r["kv_mib"]:
            r["kv_vs_baseline"] = round(base_kv / r["kv_mib"], 2)


def print_table(results: list[dict], baseline_type: str) -> None:
    print("\n" + "=" * 78)
    print(f"KV-cache quality A/B  (baseline = {baseline_type}, greedy, identical model+prompts)")
    print("=" * 78)
    head = f"{'type':<9}{'KV MiB':>9}{'shrink':>8}{'tok/s':>8}{'exact':>9}{'agree%':>9}"
    print(head)
    print("-" * len(head))
    for r in results:
        kv = f"{r['kv_mib']:.0f}" if r.get("kv_mib") else "?"
        shrink = f"{r['kv_vs_baseline']:.2f}x" if r.get("kv_vs_baseline") else ("1.00x" if r["cache_type"] == baseline_type else "?")
        tps = f"{r['mean_tok_per_s']:.1f}" if r.get("mean_tok_per_s") else "?"
        exact = f"{r.get('exact_match', '?')}/{r.get('n_prompts', '?')}"
        agree = f"{r['mean_prefix_agreement'] * 100:.1f}" if r.get("mean_prefix_agreement") is not None else "?"
        print(f"{r['cache_type']:<9}{kv:>9}{shrink:>8}{tps:>8}{exact:>9}{agree:>9}")
    print("-" * len(head))
    print("shrink = baseline KV / this KV (higher = smaller cache).  "
          "exact = prompts matching f16 verbatim.  agree% = mean char prefix kept.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to the .gguf model.")
    p.add_argument("--bin", required=True, help="Path to the (TurboQuant) llama-server binary.")
    p.add_argument("--types", default="f16,q8_0,turbo3,turbo4",
                   help="Comma-separated cache types to compare (first present f16 is the baseline).")
    p.add_argument("--baseline", default="f16", help="Cache type to score the others against.")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=999)
    p.add_argument("--n-predict", type=int, default=160)
    p.add_argument("--seed", type=int, default=20260608)
    p.add_argument("--port", type=int, default=8270)
    p.add_argument("--load-timeout", type=float, default=180.0)
    p.add_argument("--req-timeout", type=float, default=180.0)
    p.add_argument("--prompt-file", help="Optional text file; prompts separated by a line containing only '---'.")
    p.add_argument("--json-out", help="Optional path to write the full result JSON (incl. raw outputs).")
    return p.parse_args()


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    raw = Path(path).read_text(encoding="utf-8")
    parts = [chunk.strip() for chunk in re.split(r"(?m)^---\s*$", raw)]
    return [pt for pt in parts if pt]


def main() -> int:
    args = parse_args()
    if not Path(args.model).exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if not Path(args.bin).exists():
        print(f"llama-server binary not found: {args.bin}", file=sys.stderr)
        return 2
    args.prompts = load_prompts(args.prompt_file)
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    results: list[dict] = []
    for ct in types:
        try:
            results.append(run_type(args, ct))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {ct} failed: {type(exc).__name__}: {exc}", flush=True)
            results.append({"cache_type": ct, "error": f"{type(exc).__name__}: {exc}", "outputs": []})

    ok = [r for r in results if "error" not in r]
    score_against_baseline(ok, args.baseline)
    print_table(ok, args.baseline)

    failed = [r for r in results if "error" in r]
    if failed:
        print("\nfailures:")
        for r in failed:
            print(f"  {r['cache_type']}: {r['error']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
