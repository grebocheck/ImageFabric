"""Image backend built on a custom diffusers pipeline (per the chosen design —
no ComfyUI). Memory strategy is Forge-style frugal:

* SDXL (~6.6 GB) fits fully in 16 GB VRAM -> load straight to CUDA, fastest path.
* FLUX fp8 (~16 GB all-in-one) -> ``enable_model_cpu_offload`` so the text
  encoders / VAE live in RAM and only the transformer holds VRAM during denoise.

In STUB mode (the default for the foundation) no torch is touched: load/unload
just toggle, and ``generate`` renders a labelled placeholder so the queue,
arbiter swap, progress events and gallery can all be exercised today.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from ..config import settings
from ..core.enums import ModelFamily
from ..util import imaging
from .base import ImageBackend, ModelDescriptor, ProgressCb


class DiffusersImageBackend(ImageBackend):
    def __init__(self, descriptor: ModelDescriptor) -> None:
        super().__init__(descriptor)
        self._pipe: Any = None  # diffusers pipeline in real mode

    # ----------------------------------------------------------------- load
    async def load(self) -> None:
        if self._loaded:
            return
        if settings.stub_mode:
            await asyncio.sleep(0.4)  # simulate load latency
            self._loaded = True
            return
        # --- real path (exercised in M0) ---
        await asyncio.to_thread(self._load_pipeline_sync)
        self._loaded = True

    def _load_pipeline_sync(self) -> None:
        # Loaders verified on RTX 5070 Ti (Blackwell) in M0.
        import os  # noqa: PLC0415
        import re  # noqa: PLC0415

        import torch  # noqa: PLC0415  (lazy: only when GPU mode is on)

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        if self.descriptor.family is ModelFamily.FLUX:
            from diffusers import FluxPipeline  # noqa: PLC0415

            # The local file is an fp8 ComfyUI all-in-one checkpoint. Load it
            # *keeping fp8* (loading as bf16 would balloon to ~24 GB and OOM the
            # CPU during conversion). A non-gated config repo supplies the
            # pipeline config + tokenizers; weights come from the local file.
            pipe = FluxPipeline.from_single_file(
                str(self.descriptor.path),
                config=settings.flux_config_repo,
                torch_dtype=torch.float8_e4m3fn,
            )
            # Keep the big linears fp8 in VRAM, upcast per-layer to bf16 for
            # compute (Forge-style). NOTE (M0): this still materializes bf16 and
            # is slow / memory-heavy at 1024 on 16 GB — a fast FLUX needs a 4-bit
            # model (Nunchaku/GGUF) or true fp8 GEMM (torchao). Tracked for M4.
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
            )
            skip = re.compile(r"pos_embed|patch_embed|norm")
            for name, mod in pipe.transformer.named_modules():
                if skip.search(name) or name.split(".")[-1] in ("proj_in", "proj_out"):
                    mod.to(torch.bfloat16)
            pipe.text_encoder.to(torch.bfloat16)
            pipe.text_encoder_2.to(torch.bfloat16)
            pipe.vae.to(torch.bfloat16)
            pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()  # frugal VRAM: encoders idle in RAM
        else:  # SDXL — fits fully in 16 GB, keep resident & fast (~5 s / image)
            from diffusers import StableDiffusionXLPipeline  # noqa: PLC0415

            pipe = StableDiffusionXLPipeline.from_single_file(
                str(self.descriptor.path), torch_dtype=torch.float16
            )
            pipe = pipe.to("cuda")
        self._pipe = pipe

    # --------------------------------------------------------------- unload
    async def unload(self) -> None:
        if not self._loaded:
            return
        if not settings.stub_mode and self._pipe is not None:
            await asyncio.to_thread(self._free_pipeline_sync)
        self._pipe = None
        self._loaded = False

    def _free_pipeline_sync(self) -> None:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        del self._pipe
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ------------------------------------------------------------- generate
    async def generate(
        self, params: dict[str, Any], progress: ProgressCb
    ) -> list[dict[str, Any]]:
        width = int(params.get("width", settings.default_width))
        height = int(params.get("height", settings.default_height))
        steps = int(params.get("steps", settings.default_steps))
        batch = int(params.get("batch_size", 1))
        base_seed = params.get("seed")
        if base_seed in (None, -1):
            base_seed = random.randint(0, 2**31 - 1)

        results: list[dict[str, Any]] = []
        for i in range(batch):
            seed = int(base_seed) + i
            if settings.stub_mode:
                rec = await self._generate_stub(params, width, height, steps, seed, i, batch, progress)
            else:
                rec = await self._generate_real(params, width, height, steps, seed, i, batch, progress)
            results.append(rec)
        return results

    async def _generate_stub(self, params, width, height, steps, seed, i, batch, progress) -> dict[str, Any]:
        for s in range(steps):
            await asyncio.sleep(0.03)
            frac = (i + (s + 1) / steps) / batch
            await progress(frac, f"step {s + 1}/{steps} (img {i + 1}/{batch})")
        meta = {**params, "seed": seed, "width": width, "height": height,
                "model": self.descriptor.name, "stub": True}
        img = imaging.make_placeholder(width, height, [
            f"[STUB] {self.descriptor.name}",
            f"seed={seed}  {width}x{height}  steps={steps}",
            f"prompt: {params.get('prompt', '')}",
        ])
        return self._persist(img, meta, seed, width, height)

    async def _generate_real(self, params, width, height, steps, seed, i, batch, progress) -> dict[str, Any]:
        import torch  # noqa: PLC0415

        loop = asyncio.get_running_loop()

        def _step_cb(pipe, step, timestep, kw):
            frac = (i + (step + 1) / steps) / batch
            asyncio.run_coroutine_threadsafe(
                progress(frac, f"step {step + 1}/{steps} (img {i + 1}/{batch})"), loop
            )
            return kw

        def _run():
            gen = torch.Generator(device="cuda").manual_seed(seed)
            out = self._pipe(
                prompt=params.get("prompt", ""),
                negative_prompt=params.get("negative") or None,
                width=width, height=height,
                num_inference_steps=steps,
                guidance_scale=float(params.get("guidance", settings.default_guidance)),
                generator=gen,
                callback_on_step_end=_step_cb,
            )
            return out.images[0]

        img = await asyncio.to_thread(_run)
        meta = {**params, "seed": seed, "width": width, "height": height,
                "model": self.descriptor.name}
        return self._persist(img, meta, seed, width, height)

    def _persist(self, img, meta, seed, width, height) -> dict[str, Any]:
        out_dir = imaging.day_dir(settings.outputs_dir)
        stem = f"{seed}_{random.randint(1000, 9999)}"
        png_path = out_dir / f"{stem}.png"
        thumb_path = out_dir / f"{stem}.thumb.webp"
        imaging.save_png(img, png_path, meta)
        imaging.make_thumbnail(img, thumb_path)
        return {
            "path": str(png_path),
            "thumb_path": str(thumb_path),
            "seed": seed,
            "width": width,
            "height": height,
            "params": meta,
        }
