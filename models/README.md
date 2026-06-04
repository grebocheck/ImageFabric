# Models

Model weights are **not** tracked in git because they are huge. By default, keep
every local model file, model repo, and LoRA under this `models/` folder so the
app has one predictable place to scan and validate.

```text
models/
|- image/   *.safetensors or model folders (FLUX, FLUX.2 klein, SDXL, ...)
|- lora/    *.safetensors/.pt/.bin        (SDXL/FLUX LoRA adapters, SDXL turbo)
|- llm/     *.gguf                        (llama.cpp GGUF models)
|- tts/     *.gguf                        (llama-tts voice/acoustic models)
|- transcribe/ Whisper model folders/.pt  (local transcription models)
|- embed/   *.gguf embedding models       (RAG workspace)
`- vision/  *.gguf + mmproj GGUF          (Vision workspace)
```

The backend scans these folders on startup:

| Folder | Extensions / marker | Detected families |
|--------|---------------------|-------------------|
| `image/` | `.safetensors` | `flux`, `sdxl` |
| `image/<repo>/` | `model_index.json` | `flux2` |
| `lora/` | `.safetensors`, `.pt`, `.bin` | `flux`, `sdxl`, or unknown |
| `llm/` | `.gguf` | `gguf` (llama.cpp) |
| `tts/` | `.gguf` | TTS models for `llama-tts` |
| `transcribe/` | local faster-whisper folders or `.pt`/`.pth` | Whisper transcription models |
| `embed/` | `.gguf` | RAG embedding models for llama.cpp `--embeddings` |
| `vision/` | model `.gguf` + `mmproj*.gguf` | Multimodal models for `llama-mtmd-cli` |

FLUX.2 klein is a multi-file diffusers repo, not a single `.safetensors`; put the
downloaded folder under `models/image/`, for example:

```powershell
huggingface-cli download black-forest-labs/FLUX.2-klein-9B --local-dir models/image/flux2-klein-9b
```

The working local FLUX.2 klein runtime layout is:

```text
models/image/flux2-klein-9b/
|- model_index.json
|- scheduler/
|- text_encoder/
|- tokenizer/
|- transformer/
`- vae/
```

If you also keep an original-format `flux-2-klein-9b.safetensors` transformer in
`models/image/`, HFabric treats the repo folder as the runtime model and the
single file as a conversion/source artifact.

The experimental FLUX.2 nunchaku fast path uses a separate local folder:

```text
models/image/flux2-klein-9b-nunchaku/
|- svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors
|- transformer_flux2.py
`- torch_transfer_utils.py
```

Those sidecar Python files are loaded dynamically from the model folder; they are
not copied into `.venv`. The nunchaku transformer is used with the existing
`models/image/flux2-klein-9b/` diffusers repo and a bitsandbytes 4-bit Qwen3
text encoder.

On Blackwell GPUs, FLUX.2 nunchaku int4 is kept as a local file if downloaded
but is hidden from the runtime model list because nunchaku requires fp4 for this
GPU family.

Environment variables like `HFAB_IMAGE_MODELS_DIR`, `HFAB_LORA_MODELS_DIR`,
`HFAB_LLM_MODELS_DIR`, and `HFAB_TTS_MODELS_DIR` exist for development, but
the project default is to keep model storage inside `models/`.

TTS output WAV files and their JSON sidecars are runtime artifacts, so they are
written under `data/outputs/<date>/`, not under `models/`.
