# Models

Model weights are **not** tracked in git because they are huge. By default, keep
every local model file, model repo, and LoRA under this `models/` folder so the
app has one predictable place to scan and validate.

```text
models/
|- image/   *.safetensors or model folders (FLUX, FLUX.2 klein, SDXL, ...)
|- lora/    *.safetensors/.pt/.bin        (SDXL/FLUX LoRA adapters)
|- llm/     *.gguf                        (llama.cpp GGUF models)
`- tts/     *.gguf                        (llama-tts voice/acoustic models)
```

The backend scans these folders on startup:

| Folder | Extensions / marker | Detected families |
|--------|---------------------|-------------------|
| `image/` | `.safetensors` | `flux`, `sdxl` |
| `image/<repo>/` | `model_index.json` | `flux2` |
| `lora/` | `.safetensors`, `.pt`, `.bin` | `flux`, `sdxl`, or unknown |
| `llm/` | `.gguf` | `gguf` (llama.cpp) |
| `tts/` | `.gguf` | TTS models for `llama-tts` |

FLUX.2 klein is a multi-file diffusers repo, not a single `.safetensors`; put the
downloaded folder under `models/image/`, for example:

```powershell
huggingface-cli download black-forest-labs/FLUX.2-klein-9B --local-dir models/image/flux2-klein-9b
```

Environment variables like `IMGFAB_IMAGE_MODELS_DIR`, `IMGFAB_LORA_MODELS_DIR`,
`IMGFAB_LLM_MODELS_DIR`, and `IMGFAB_TTS_MODELS_DIR` exist for development, but
the project default is to keep model storage inside `models/`.

TTS output WAV files and their JSON sidecars are runtime artifacts, so they are
written under `data/outputs/<date>/`, not under `models/`.
