# Models

Model weights are **not** tracked in git (they are huge). Drop your files here:

```text
models/
├─ image/   *.safetensors          (FLUX, SDXL, ... checkpoints)
├─ lora/    *.safetensors/.pt/.bin (SDXL/FLUX LoRA adapters)
└─ llm/     *.gguf                 (llama.cpp GGUF models)
```

The backend scans these folders on startup (reading only the safetensors
*header*, so it is instant) and classifies each file automatically:

| Folder | Extensions | Detected families |
|--------|------------|-------------------|
| `image/` | `.safetensors` | `flux`, `sdxl` |
| `lora/` | `.safetensors`, `.pt`, `.bin` | `flux`, `sdxl`, or unknown |
| `llm/` | `.gguf` | `gguf` (llama.cpp) |

Override the locations with `IMGFAB_IMAGE_MODELS_DIR`, `IMGFAB_LORA_MODELS_DIR`,
or `IMGFAB_LLM_MODELS_DIR` if you keep weights elsewhere (e.g. a different
drive).
