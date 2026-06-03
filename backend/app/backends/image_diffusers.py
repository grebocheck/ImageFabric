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
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
import random
from typing import Any

from ..config import settings
from ..core.enums import ModelFamily
from ..util import imaging, sysmon
from .base import ImageBackend, ModelDescriptor, ProgressCb


class DiffusersImageBackend(ImageBackend):
    def __init__(self, descriptor: ModelDescriptor) -> None:
        super().__init__(descriptor)
        self._pipe: Any = None  # diffusers pipeline in real mode
        self._active_features: dict[str, Any] = {}
        self._loaded_loras: dict[str, str] = {}

    @property
    def can_keep_warm(self) -> bool:
        return True

    # ----------------------------------------------------------------- load
    async def load(self) -> None:
        if self._loaded:
            return
        if self._warm:
            if settings.stub_mode:
                await asyncio.sleep(0.1)
                self._loaded = True
                self._warm = False
                self._load_report = {"keep_warm": {"resumed": True, "stub": True}}
                return
            if self._pipe is not None:
                await asyncio.to_thread(self._resume_pipeline_sync)
                self._loaded = True
                self._warm = False
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
        self._active_features = {}
        self._loaded_loras = {}
        report: dict[str, Any] = {
            "acceleration": {},
            "memory": {"start": self._memory_snapshot(torch)},
        }

        if self.descriptor.family is ModelFamily.FLUX2:
            pipe = self._load_flux2_klein(torch)
        elif self._is_nunchaku_quant():
            pipe = self._load_nunchaku_flux(torch)
        elif self.descriptor.family is ModelFamily.FLUX:
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
        self._apply_acceleration(torch, pipe, report)
        report["memory"]["end"] = self._memory_snapshot(torch)
        self._load_report = report

    def _load_nunchaku_flux(self, torch) -> Any:
        """SVDQuant fp4 FLUX (Blackwell turbo): ~8 s/1024, peak RAM ~13 GB.

        Assembled from light components so a single load stays well within the RAM
        budget (P0.1): nunchaku fp4 transformer + int4 T5 (~3 GB, not ~10 GB bf16),
        with CLIP/VAE/tokenizers/scheduler from the non-gated config repo. We do
        NOT read the local 16 GB fp8 checkpoint here."""
        from diffusers import FluxPipeline  # noqa: PLC0415
        from nunchaku import (  # noqa: PLC0415
            NunchakuFluxTransformer2dModel,
            NunchakuT5EncoderModel,
        )

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(str(self.descriptor.path))
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(settings.flux_t5_nunchaku)
        pipe = FluxPipeline.from_pretrained(
            settings.flux_config_repo,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
        return pipe

    def _load_flux2_klein(self, torch) -> Any:
        """FLUX.2 [klein] via diffusers (nunchaku has no FLUX.2 transformer yet).

        klein's text encoder is a small Qwen3 (not FLUX.2 [dev]'s 24 GB Mistral),
        so the 9B model in bitsandbytes 4-bit + model-offload fits a 16 GB card.
        The model is a multi-file repo folder under models/image/."""
        from diffusers import Flux2KleinPipeline  # noqa: PLC0415

        source = str(self.descriptor.path)
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}

        quant = settings.flux2_quant.lower().strip()
        if quant in ("bnb-nf4", "bnb-fp4"):
            from diffusers import PipelineQuantizationConfig  # noqa: PLC0415

            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4" if quant == "bnb-nf4" else "fp4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )
        elif quant not in ("", "none", "bf16"):
            raise ValueError(
                "IMGFAB_FLUX2_QUANT must be one of: bnb-nf4, bnb-fp4, none "
                f"(got {settings.flux2_quant!r})"
            )

        pipe = Flux2KleinPipeline.from_pretrained(source, **kwargs)
        if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
            pipe.vae.enable_tiling()

        offload = settings.flux2_offload.lower().strip()
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload in ("", "none"):
            pipe.to("cuda")
        else:  # "model" (default): encoders idle in RAM, frugal VRAM
            pipe.enable_model_cpu_offload()
        return pipe

    def _is_nunchaku_quant(self) -> bool:
        return bool(self.descriptor.quant and self.descriptor.quant.startswith("nunchaku"))

    def _apply_acceleration(self, torch, pipe: Any, report: dict[str, Any]) -> None:
        self._configure_attention(torch, report)
        self._maybe_apply_flux_step_cache(pipe, report)
        self._maybe_apply_sdxl_turbo_lora(pipe, report)
        self._maybe_compile_transformer(torch, pipe, report)

    def _configure_attention(self, torch, report: dict[str, Any]) -> None:
        mode = settings.attention_backend.lower().strip() or "auto"
        precision = settings.attention_matmul_precision.lower().strip()
        if precision not in ("highest", "high", "medium"):
            raise ValueError(
                "IMGFAB_ATTENTION_MATMUL_PRECISION must be one of: highest, high, medium "
                f"(got {settings.attention_matmul_precision!r})"
            )
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(precision)

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                matmul.allow_tf32 = settings.attention_allow_tf32

        attention = getattr(torch.nn, "attention", None)
        sdp_backend = getattr(attention, "SDPBackend", None) if attention else None
        sdpa_kernel = getattr(attention, "sdpa_kernel", None) if attention else None
        native_backends = self._native_sdp_backend_names(sdp_backend)
        backend_map = {
            "flash": "FLASH_ATTENTION",
            "efficient": "EFFICIENT_ATTENTION",
            "math": "MATH",
            "cudnn": "CUDNN_ATTENTION",
        }
        aliases = {"default": "auto", "sdpa": "auto", "native": "auto"}
        mode = aliases.get(mode, mode)
        if mode not in {"auto", *backend_map}:
            raise ValueError(
                "IMGFAB_ATTENTION_BACKEND must be one of: auto, flash, efficient, math, cudnn "
                f"(got {settings.attention_backend!r})"
            )
        if mode != "auto":
            enum_name = backend_map[mode]
            if sdpa_kernel is None or sdp_backend is None or enum_name not in native_backends:
                raise ValueError(
                    f"IMGFAB_ATTENTION_BACKEND={mode!r} requires torch.nn.attention."
                    f"SDPBackend.{enum_name}, but this torch build does not expose it."
                )
            if mode in {"flash", "efficient", "cudnn"} and not cuda_available:
                raise ValueError(
                    f"IMGFAB_ATTENTION_BACKEND={mode!r} requires a CUDA torch build/device."
                )

        feature = {
            "requested": settings.attention_backend,
            "mode": mode,
            "native_sdpa": sdpa_kernel is not None and sdp_backend is not None,
            "native_backends": native_backends,
            "forced_backend": backend_map.get(mode),
            "cuda_available": cuda_available,
            "external_flash_attn": find_spec("flash_attn") is not None,
            "xformers": find_spec("xformers") is not None,
            "float8_dtypes": self._torch_float8_dtypes(torch),
            "allow_tf32": settings.attention_allow_tf32,
            "matmul_precision": precision,
        }
        self._active_features["attention"] = feature
        report["acceleration"]["attention"] = feature

    @staticmethod
    def _native_sdp_backend_names(sdp_backend: Any) -> list[str]:
        if sdp_backend is None:
            return []
        return [
            name
            for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION")
            if hasattr(sdp_backend, name)
        ]

    @staticmethod
    def _torch_float8_dtypes(torch) -> list[str]:
        return sorted(name for name in dir(torch) if name.startswith("float8_"))

    def _maybe_apply_flux_step_cache(self, pipe: Any, report: dict[str, Any]) -> None:
        if self.descriptor.family is not ModelFamily.FLUX:
            return

        mode = settings.flux_step_cache.lower().strip()
        if mode in ("", "off", "none", "false"):
            return
        if mode == "fb":
            from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_pipe  # noqa: PLC0415

            apply_cache_on_pipe(
                pipe,
                residual_diff_threshold=settings.flux_fb_cache_threshold,
                use_double_fb_cache=settings.flux_fb_cache_double,
            )
            self._active_features["flux_step_cache"] = {
                "mode": "fb",
                "threshold": settings.flux_fb_cache_threshold,
                "double": settings.flux_fb_cache_double,
            }
        elif mode == "teacache":
            self._active_features["flux_step_cache"] = {
                "mode": "teacache",
                "threshold": settings.flux_teacache_threshold,
                "skip_steps": settings.flux_teacache_skip_steps,
            }
        else:
            raise ValueError(
                "IMGFAB_FLUX_STEP_CACHE must be one of: off, fb, teacache "
                f"(got {settings.flux_step_cache!r})"
            )
        report["acceleration"]["flux_step_cache"] = self._active_features["flux_step_cache"]

    def _maybe_apply_sdxl_turbo_lora(self, pipe: Any, report: dict[str, Any]) -> None:
        if self.descriptor.family is not ModelFamily.SDXL or not settings.sdxl_turbo_lora:
            return

        source = settings.sdxl_turbo_lora
        path = Path(source)
        if path.suffix.lower() == ".safetensors":
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name="turbo")
        else:
            pipe.load_lora_weights(source, adapter_name="turbo")
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["turbo"], adapter_weights=[settings.sdxl_turbo_lora_weight])

        self._active_features["sdxl_turbo_lora"] = {
            "source": source,
            "weight": settings.sdxl_turbo_lora_weight,
            "default_steps": settings.sdxl_turbo_steps,
            "default_guidance": settings.sdxl_turbo_guidance,
        }
        report["acceleration"]["sdxl_turbo_lora"] = self._active_features["sdxl_turbo_lora"]

    def _maybe_compile_transformer(self, torch, pipe: Any, report: dict[str, Any]) -> None:
        if not settings.torch_compile:
            return
        if not hasattr(pipe, "transformer"):
            report["acceleration"]["torch_compile"] = {"skipped": "pipeline has no transformer"}
            return

        compile_report: dict[str, Any] = {
            "mode": settings.torch_compile_mode,
            "before": self._memory_snapshot(torch),
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pipe.transformer = torch.compile(pipe.transformer, mode=settings.torch_compile_mode)
        compile_report["after_wrap"] = self._memory_snapshot(torch)

        if settings.torch_compile_warmup:
            self._warmup_pipeline(torch, pipe)
            compile_report["after_warmup"] = self._memory_snapshot(torch)
            if torch.cuda.is_available():
                compile_report["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                compile_report["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)

        self._active_features["torch_compile"] = {"mode": settings.torch_compile_mode}
        report["acceleration"]["torch_compile"] = compile_report

    def _warmup_pipeline(self, torch, pipe: Any) -> None:
        size = int(settings.torch_compile_warmup_size)
        size = max(256, (size // 64) * 64)
        kwargs = {
            "prompt": "warmup",
            "width": size,
            "height": size,
            "num_inference_steps": 1,
            "guidance_scale": settings.default_guidance,
            "generator": torch.Generator(device="cuda").manual_seed(0),
        }
        with torch.inference_mode(), self._attention_context(torch):
            pipe(**kwargs)

    def _memory_snapshot(self, torch) -> dict[str, Any]:
        snap = sysmon.snapshot()
        if torch.cuda.is_available():
            snap["cuda_process"] = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            }
        return snap

    # --------------------------------------------------------------- unload
    async def unload(self) -> None:
        if not self._loaded and not self._warm:
            return
        if not settings.stub_mode and self._pipe is not None:
            await asyncio.to_thread(self._free_pipeline_sync)
        self._pipe = None
        self._loaded = False
        self._warm = False
        self._active_features = {}
        self._loaded_loras = {}

    async def park(self) -> bool:
        if not self._loaded:
            return False
        if settings.stub_mode:
            await asyncio.sleep(0.1)
            self._loaded = False
            self._warm = True
            return True
        if self._pipe is None:
            return False
        await asyncio.to_thread(self._park_pipeline_sync)
        self._loaded = False
        self._warm = True
        return True

    def _park_pipeline_sync(self) -> None:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if hasattr(self._pipe, "maybe_free_model_hooks"):
            self._pipe.maybe_free_model_hooks()
        if self.descriptor.family is ModelFamily.SDXL and hasattr(self._pipe, "to"):
            self._pipe.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _resume_pipeline_sync(self) -> None:
        import torch  # noqa: PLC0415

        report: dict[str, Any] = {
            "keep_warm": {"resumed": True},
            "memory": {"start": self._memory_snapshot(torch)},
        }
        if self.descriptor.family is ModelFamily.SDXL and hasattr(self._pipe, "to"):
            self._pipe.to("cuda")
        elif hasattr(self._pipe, "enable_model_cpu_offload"):
            self._pipe.enable_model_cpu_offload()
        report["memory"]["end"] = self._memory_snapshot(torch)
        self._load_report = report

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
        steps = self._steps(params)
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
        meta = {**self._public_params(params), "seed": seed, "width": width, "height": height,
                "model": self.descriptor.name, "stub": True,
                "acceleration": self._active_features}
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
            self._apply_runtime_loras(params)
            with self._generation_context(steps), self._attention_context(torch):
                out = self._pipe(
                    prompt=params.get("prompt", ""),
                    negative_prompt=params.get("negative") or None,
                    width=width, height=height,
                    num_inference_steps=steps,
                    guidance_scale=self._guidance(params),
                    generator=gen,
                    callback_on_step_end=_step_cb,
                )
            return out.images[0]

        img = await asyncio.to_thread(_run)
        meta = {**self._public_params(params), "seed": seed, "width": width, "height": height,
                "steps": steps, "guidance": self._guidance(params),
                "model": self.descriptor.name, "acceleration": self._active_features}
        return self._persist(img, meta, seed, width, height)

    def _steps(self, params: dict[str, Any]) -> int:
        steps = int(params.get("steps", settings.default_steps))
        untouched = "steps" not in params or steps == settings.default_steps
        if self.descriptor.family is ModelFamily.FLUX2 and untouched:
            return settings.flux2_default_steps
        if self._active_features.get("sdxl_turbo_lora") and params.get("turbo", True):
            if untouched:
                return settings.sdxl_turbo_steps
        return steps

    def _guidance(self, params: dict[str, Any]) -> float:
        guidance = float(params.get("guidance", settings.default_guidance))
        untouched = "guidance" not in params or guidance == settings.default_guidance
        if self.descriptor.family is ModelFamily.FLUX2 and untouched:
            return settings.flux2_default_guidance
        if self._active_features.get("sdxl_turbo_lora") and params.get("turbo", True):
            if untouched:
                return settings.sdxl_turbo_guidance
        return guidance

    def _generation_context(self, steps: int):
        feature = self._active_features.get("flux_step_cache")
        if not feature or feature.get("mode") != "teacache":
            return nullcontext()

        from nunchaku.caching.teacache import TeaCache  # noqa: PLC0415

        return TeaCache(
            self._pipe.transformer,
            num_steps=steps,
            rel_l1_thresh=settings.flux_teacache_threshold,
            skip_steps=settings.flux_teacache_skip_steps,
            enabled=True,
            model_name="flux",
        )

    def _attention_context(self, torch):
        forced_backend = (self._active_features.get("attention") or {}).get("forced_backend")
        if not forced_backend:
            return nullcontext()

        attention = getattr(torch.nn, "attention", None)
        if attention is None:
            return nullcontext()
        sdpa_kernel = getattr(attention, "sdpa_kernel", None)
        sdp_backend = getattr(attention, "SDPBackend", None)
        if sdpa_kernel is None or sdp_backend is None or not hasattr(sdp_backend, forced_backend):
            return nullcontext()
        return sdpa_kernel([getattr(sdp_backend, forced_backend)])

    def _apply_runtime_loras(self, params: dict[str, Any]) -> None:
        adapters: list[str] = []
        weights: list[float] = []
        turbo = self._active_features.get("sdxl_turbo_lora")
        if turbo and params.get("turbo", True):
            adapters.append("turbo")
            weights.append(float(turbo["weight"]))

        requests = self._lora_requests(params)
        if requests and not hasattr(self._pipe, "load_lora_weights"):
            raise RuntimeError(f"Pipeline for {self.descriptor.name} does not support LoRA loading")
        for request in requests:
            adapter = self._load_lora_adapter(request["id"], Path(request["path"]))
            adapters.append(adapter)
            weights.append(float(request["weight"]))

        if adapters:
            if not hasattr(self._pipe, "set_adapters"):
                raise RuntimeError(f"Pipeline for {self.descriptor.name} does not support LoRA adapters")
            self._pipe.set_adapters(adapters, adapter_weights=weights)
            if hasattr(self._pipe, "enable_lora"):
                self._pipe.enable_lora()
        elif hasattr(self._pipe, "disable_lora"):
            self._pipe.disable_lora()

    def _lora_requests(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        raw_loras = params.get("loras") or []
        paths = params.get("_lora_paths") or {}
        if not isinstance(raw_loras, list) or not isinstance(paths, dict):
            return []

        requests: list[dict[str, Any]] = []
        for item in raw_loras:
            if not isinstance(item, dict):
                continue
            lora_id = item.get("id")
            if not isinstance(lora_id, str):
                continue
            path = item.get("path") or paths.get(lora_id)
            if not isinstance(path, str):
                raise RuntimeError(f"LoRA {lora_id!r} is missing its validated path")
            requests.append({
                "id": lora_id,
                "path": path,
                "weight": float(item.get("weight", 1.0)),
            })
        return requests

    def _load_lora_adapter(self, lora_id: str, path: Path) -> str:
        if lora_id in self._loaded_loras:
            return self._loaded_loras[lora_id]
        if not path.exists():
            raise FileNotFoundError(f"LoRA file not found: {path}")

        adapter = self._lora_adapter_name(lora_id)
        if path.is_dir():
            self._pipe.load_lora_weights(str(path), adapter_name=adapter)
        elif path.suffix.lower() == ".safetensors":
            self._pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
        else:
            self._pipe.load_lora_weights(str(path), adapter_name=adapter)
        self._loaded_loras[lora_id] = adapter
        return adapter

    @staticmethod
    def _lora_adapter_name(lora_id: str) -> str:
        body = "".join(ch if ch.isalnum() else "_" for ch in lora_id)[:80]
        return f"lora_{body}"

    @staticmethod
    def _public_params(params: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in params.items() if not k.startswith("_")}

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
