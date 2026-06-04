"""Local llama.cpp embeddings for RAG.

The service starts a dedicated llama-server in embeddings mode on demand. It is
CPU-only by default, so RAG indexing/search does not compete with the GPU
arbiter that owns image and chat models.
"""

from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path
from typing import Any

import httpx

from ..config import settings


def _model_id(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def list_embedding_models() -> list[dict[str, Any]]:
    root = settings.embed_models_dir
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*.gguf")):
        out.append({
            "id": _model_id(path),
            "name": path.stem,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        })
    return out


def embedding_model_map() -> dict[str, dict[str, Any]]:
    return {m["id"]: m for m in list_embedding_models()}


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


class LocalEmbeddingService:
    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._model_id: str | None = None
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return f"http://{settings.llama_host}:{settings.llama_embed_port}"

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def embed(self, texts: list[str], model_id: str | None = None) -> list[list[float]]:
        clean = [text.strip() for text in texts if text.strip()]
        if not clean:
            return []
        models = embedding_model_map()
        if not models:
            raise RuntimeError(f"no embedding models found in {settings.embed_models_dir}")
        model = models.get(model_id or "") or next(iter(models.values()))

        async with self._lock:
            await self._ensure_server_locked(model)
            payload = {"model": model["name"], "input": clean}
            async with httpx.AsyncClient(timeout=settings.embed_timeout_seconds) as client:
                resp = await client.post(f"{self.base_url}/v1/embeddings", json=payload)
                resp.raise_for_status()
                data = resp.json().get("data", [])
        vectors = [_normalize([float(x) for x in item["embedding"]]) for item in data]
        if len(vectors) != len(clean):
            raise RuntimeError("embedding server returned an unexpected number of vectors")
        return vectors

    async def _ensure_server_locked(self, model: dict[str, Any]) -> None:
        if self._proc and self._proc.returncode is None and self._model_id == model["id"]:
            return
        await self._stop_locked()
        if not settings.llama_server_bin.exists():
            raise FileNotFoundError(f"llama-server binary not found at {settings.llama_server_bin}")

        self._proc = await asyncio.create_subprocess_exec(
            str(settings.llama_server_bin),
            "-m",
            str(model["path"]),
            "--host",
            settings.llama_host,
            "--port",
            str(settings.llama_embed_port),
            "--embeddings",
            "-ngl",
            str(settings.embed_gpu_layers),
            "--pooling",
            "mean",
            "--no-warmup",
            cwd=str(settings.llama_server_bin.parent),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._model_id = model["id"]
        await self._wait_healthy_locked()

    async def _wait_healthy_locked(self) -> None:
        deadline = asyncio.get_running_loop().time() + settings.embed_timeout_seconds
        async with httpx.AsyncClient() as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._proc and self._proc.returncode is not None:
                    raise RuntimeError("embedding llama-server exited during startup")
                try:
                    resp = await client.get(f"{self.base_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.4)
        raise TimeoutError("embedding llama-server did not become healthy in time")

    async def _stop_locked(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        self._model_id = None


embedding_service = LocalEmbeddingService()
