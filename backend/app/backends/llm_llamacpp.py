"""LLM backend that drives a `llama-server` subprocess (llama.cpp).

Running it as a separate process is deliberate: when we need VRAM for image
generation we terminate the server process, which releases llama.cpp's VRAM
completely. ``load`` starts the server and waits for ``/health``; ``unload``
terminates it, escalating to kill only if shutdown hangs.

STUB mode simulates a streamed completion (a deterministic prompt expansion) so
the LLM->image phase pipeline works without a real model or binary.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..config import settings
from .base import LLMBackend, ModelDescriptor, TokenCb


class LlamaCppBackend(LLMBackend):
    def __init__(self, descriptor: ModelDescriptor) -> None:
        super().__init__(descriptor)
        self._proc: asyncio.subprocess.Process | None = None
        self._stop = False

    def request_stop(self) -> None:
        """Best-effort interrupt of the in-flight streamed completion."""
        self._stop = True

    @property
    def _base_url(self) -> str:
        return f"http://{settings.llama_host}:{settings.llama_port}"

    # ----------------------------------------------------------------- load
    async def load(self) -> None:
        if self._loaded:
            return
        if settings.stub_mode:
            await asyncio.sleep(0.3)
            self._loaded = True
            return
        await self._start_server()
        self._loaded = True

    async def _start_server(self) -> None:
        if not settings.llama_server_bin.exists():
            raise FileNotFoundError(
                f"llama-server binary not found at {settings.llama_server_bin}. "
                "Place a CUDA(sm_120) build there or set HFAB_LLAMA_SERVER_BIN."
            )
        # Launch with cwd = the binary's folder so ggml's dynamic CUDA backend
        # (ggml-cuda.dll, cudart, cublas — all shipped beside llama-server.exe) is
        # found. Otherwise ggml silently falls back to CPU, which both eats ~12 GB
        # of RAM and is far slower.
        self._proc = await asyncio.create_subprocess_exec(
            str(settings.llama_server_bin),
            "-m", str(self.descriptor.path),
            "--host", settings.llama_host,
            "--port", str(settings.llama_port),
            "-ngl", str(settings.llama_ngl),
            "-c", str(settings.llama_ctx),
            # Disable llama's auto VRAM-fitting and honor our explicit -ngl. The
            # auto-fit probe hangs / under-offloads when the parent process holds
            # a torch CUDA context (which it does once diffusers / the mem monitor
            # has touched CUDA) -> without this the LLM silently runs on CPU,
            # eating ~12 GB RAM and crawling.
            "--fit", "off",
            cwd=str(settings.llama_server_bin.parent),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await self._wait_healthy()

    async def _wait_healthy(self, timeout: float = 120.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        async with httpx.AsyncClient() as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._proc and self._proc.returncode is not None:
                    raise RuntimeError("llama-server exited during startup")
                try:
                    r = await client.get(f"{self._base_url}/health", timeout=2.0)
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError("llama-server did not become healthy in time")

    # --------------------------------------------------------------- unload
    async def unload(self) -> None:
        if not self._loaded:
            return
        if not settings.stub_mode and self._proc is not None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10.0)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        self._loaded = False

    # ------------------------------------------------------------- complete
    async def complete(self, params: dict[str, Any], on_token: TokenCb | None = None) -> str:
        self._stop = False
        messages = self._build_messages(params)
        if settings.stub_mode:
            return await self._complete_stub(messages, on_token)
        return await self._complete_real(messages, params, on_token)

    @staticmethod
    def _build_messages(params: dict[str, Any]) -> list[dict[str, str]]:
        if params.get("messages"):
            return params["messages"]
        msgs: list[dict[str, str]] = []
        if params.get("system"):
            msgs.append({"role": "system", "content": params["system"]})
        msgs.append({"role": "user", "content": params.get("prompt", "")})
        return msgs

    async def _complete_stub(self, messages, on_token) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = (
            f"masterpiece, best quality, highly detailed, cinematic lighting, "
            f"{user.strip()}, intricate background, 8k, sharp focus"
        )
        out = []
        for tok in text.split(" "):
            if self._stop:
                break
            await asyncio.sleep(0.02)
            piece = tok + " "
            out.append(piece)
            if on_token:
                await on_token(piece)
        return "".join(out).strip()

    async def _complete_real(self, messages, params, on_token) -> str:
        payload = {
            "model": self.descriptor.name,
            "messages": messages,
            "temperature": float(params.get("temperature", 0.8)),
            "max_tokens": int(params.get("max_tokens", 512)),
            "stream": True,
        }
        # Optional sampling knobs — pass through to llama-server when supplied.
        for key in ("top_p", "min_p", "repeat_penalty", "seed"):
            if params.get(key) is not None:
                payload[key] = params[key]
        if params.get("top_k") is not None:
            payload["top_k"] = int(params["top_k"])
        if params.get("stop"):
            payload["stop"] = params["stop"]
        import json  # noqa: PLC0415

        chunks: list[str] = []

        async def emit(piece: str) -> None:
            chunks.append(piece)
            if on_token:
                await on_token(piece)

        # Harmony models (gpt-oss) don't wrap their chain-of-thought in <think>
        # tags; llama-server surfaces it in a separate `reasoning_content` delta
        # field while the user-facing answer streams in `content`. We re-wrap the
        # reasoning in <think>…</think> so the frontend's existing reasoning split
        # (Thinking.tsx) renders it in the collapsible panel — same as DeepSeek-R1
        # / Qwen, which emit the tags inline themselves.
        in_reasoning = False
        reasoning_closed = False
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{self._base_url}/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if self._stop:
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        delta = json.loads(data)["choices"][0]["delta"]
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue

                    reasoning = delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""

                    if reasoning and not reasoning_closed:
                        if not in_reasoning:
                            in_reasoning = True
                            await emit("<think>")
                        await emit(reasoning)
                    if content:
                        if in_reasoning and not reasoning_closed:
                            reasoning_closed = True
                            await emit("</think>")
                        await emit(content)

        # Reasoning with no following answer (e.g. cut short) — close the block so
        # the frontend doesn't treat the whole reply as still-active reasoning.
        if in_reasoning and not reasoning_closed:
            await emit("</think>")
        return "".join(chunks).strip()
