# Configuration Management Guide

Unified reference for tuning and validating the GenAI RAG system. The configuration layer powers every deploy—from a local laptop to GPU-backed image generation—without code changes.

## 🧭 Overview

- Centralised in `config.py` with Pydantic-style validation, range checks, and path assertions.
- Values sourced from environment variables or a project-level `.env` file (copy `.env.example`).
- Sensitive values (API keys, Hugging Face tokens) are automatically masked in logs.
- Companion scripts automate the heavy lifting:
    1. `setup.ps1` → baseline environment + `.env` bootstrap.
    2. `setup_improvements.ps1` → applies retrieval quality upgrades (mpnet embeddings, reranking).
    3. `setup_local_image_gen.ps1` → installs PyTorch, downloads image models, and configures GPU usage.

## ⚡ Quick Start

```powershell
# 1. Initialise local config
Copy-Item .env.example .env

# 2. (Optional) apply tuned defaults and rebuild index
./setup_improvements.ps1

# 3. (Optional) prepare local Stable Diffusion / Qwen models
./setup_local_image_gen.ps1

# 4. Validate everything in one shot
uv run python config.py
```

The checker prints current values, masks secrets, confirms file paths, and warns if numeric ranges are violated.

## 📚 Configuration Reference

### Data & Paths

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_PATH` | `data/arxiv_2.9k.jsonl` | JSONL corpus used for indexing |
| `INDEX_DIR` | `index` | Location for FAISS index, metadata, hash cache |
| `MODEL_PATH` | `models/llama-model.gguf` | Phi-3 Mini (GGUF) inference file |

> Docker deployments swap these for container mounts, e.g. `/data`, `/index`, `/models`.

### Embedding Model

| Variable | Default | Notes |
|----------|---------|-------|
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | 768-dim, high-precision encoder (recommended) |
| `EMBEDDING_BATCH_SIZE` | `32` | Trade speed vs memory when building the index |

Common alternatives:
- `all-MiniLM-L6-v2` → faster, 384-dim (legacy baseline).
- `paraphrase-multilingual-mpnet-base-v2` → multilingual coverage.

### LLM Runtime

| Variable | Default | Use |
|----------|---------|-----|
| `N_THREADS` | `4` | llama.cpp inference threads |
| `N_CTX` | `4096` | Context window in tokens |
| `N_GPU_LAYERS` | `0` | Offloaded transformer layers (0 = CPU only) |

Guidance:
- CPU boxes: `N_THREADS=8` on 8-core hosts, leave `N_GPU_LAYERS=0`.
- Mixed CPU/GPU: set `N_GPU_LAYERS` to 20–35 depending on VRAM.
- Memory constrained: reduce `N_CTX` to 2048.

### Retrieval Controls

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_TOP_K` | `5` | Documents returned to the LLM |
| `MAX_TOP_K` | `20` | Safety cap enforced by API schema |
| `RERANK_MULTIPLIER` | `3` | How many candidates to gather before reranking |

The mpnet + reranker combo is tuned for factual responses. Reduce `RERANK_MULTIPLIER` to `2` for marginally faster queries.

### Generation Parameters

| Variable | Default | Behaviour |
|----------|---------|-----------|
| `MAX_TOKENS` | `600` | Hard stop for Phi-3 output |
| `TEMPERATURE` | `0.3` | Controls creativity—lowered to fight hallucinations |
| `TOP_P` | `0.85` | Nucleus sampling cutoff |
| `REPEAT_PENALTY` | `1.1` | Penalises repeated phrases |

Recommended presets:
- **Factual Q&A**: leave defaults (0.3 / 0.85 / 600).
- **Descriptive Summaries**: `TEMPERATURE=0.4`, `MAX_TOKENS=800`.
- **Creative mode**: `TEMPERATURE=0.7`, `TOP_P=0.95` (watch for hallucinations).

### Image Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_API_PROVIDER` | `local` | `local`, `pollinations`, or `openai` |
| `IMAGE_MODEL_PATH` | `models/sdxl-turbo` | Cache directory for local models |
| `IMAGE_MODEL_NAME` | `stabilityai/sdxl-turbo` | Hugging Face model identifier |
| `IMAGE_API_KEY` | *empty* | Required only for OpenAI DALL·E |

Provider guide:
1. **Local (default)** – Offline Stable Diffusion 3.5, SDXL Turbo/Base, or Qwen Lightning. Requires 10–15 GB disk and 8–12 GB VRAM. Authenticate gated models with `HF_TOKEN`.
2. **Pollinations** – Free, zero setup. Set `IMAGE_API_PROVIDER=pollinations` to skip heavyweight downloads.
3. **OpenAI** – Premium quality. Set provider to `openai` and export `IMAGE_API_KEY` before starting the app.

Optional extras:
- `HF_TOKEN` – Hugging Face access token for gated Stable Diffusion models (export or add to `.env`).
- `CUDA_VISIBLE_DEVICES=-1` – Force CPU image generation when GPU memory is limited.

### Server & Telemetry

| Variable | Default | Notes |
|----------|---------|-------|
| `HOST` | `0.0.0.0` | Bind address (`127.0.0.1` for local-only) |
| `PORT` | `8000` | Override when the default is occupied |
| `LOG_LEVEL` | `info` | `debug` during development, `warning` in production |
| `ENABLE_CORS` | `true` | Toggle cross-origin access for custom frontends |

### Performance & Resilience

| Variable | Default | Purpose |
|----------|---------|---------|
| `FORCE_REBUILD` | `false` | Skip hash check and rebuild index on startup |
| `CACHE_EMBEDDINGS` | `true` | Persist embeddings alongside FAISS index |
| `ENABLE_STREAMING` | `true` | Turn off if clients cannot consume SSE |
| `REQUEST_TIMEOUT` | `120` | Hard limit per request (increase for slow GPUs) |

### LaTeX Support

| Variable | Default | Description |
|----------|---------|-------------|
| `LATEX_PROCESSING_ENABLED` | `true` | Enables math-aware preprocessing and MathJax UI rendering |

Leave enabled for research corpora; disable only if the corpus contains literal dollar-sign usage that should remain untouched.

## 🧩 Environment Profiles

### Local Development (CPU)
```env
DATA_PATH=data/arxiv_2.9k.jsonl
INDEX_DIR=index
MODEL_PATH=models/llama-model.gguf
LOG_LEVEL=debug
N_THREADS=8
IMAGE_API_PROVIDER=pollinations
```

### Docker / Production
```env
DATA_PATH=/data/arxiv_2.9k.jsonl
INDEX_DIR=/index
MODEL_PATH=/models/llama-model.gguf
HOST=0.0.0.0
PORT=8080
LOG_LEVEL=info
ENABLE_STREAMING=true
```

### GPU Workstation with Local Stable Diffusion 3.5
```env
IMAGE_API_PROVIDER=local
IMAGE_MODEL_NAME=stabilityai/stable-diffusion-3.5-medium
IMAGE_MODEL_PATH=models/stable-diffusion-3.5-medium
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
N_GPU_LAYERS=20
REQUEST_TIMEOUT=300
```

### Pollinations Fallback (no GPU, no downloads)
```env
IMAGE_API_PROVIDER=pollinations
CACHE_EMBEDDINGS=true
FORCE_REBUILD=false
```

## 🛠️ Validation & Tooling

- `uv run python config.py` – prints the config and validates ranges/paths.
- `evaluate_rag.py` – proves the tuned configuration achieves ~78/100 evaluation score.
- `setup_improvements.ps1` – applies new defaults, wipes stale indices, triggers mpnet rebuild.
- `setup_local_image_gen.ps1` – installs PyTorch CUDA wheels, downloads models, updates `.env` with appropriate provider settings.
- `tools/test_image_gen.py` – sanity-checks GPU availability, VRAM, and sample generation.

## 🔒 Security Practices

- Never commit `.env` files; rely on `.env.example` for documentation.
- Store secrets in environment variables or secret managers (Azure Key Vault, AWS SSM) in production deployments.
- Mask tokens in logs—`config.print_config()` already obscures sensitive suffixes.
- Use unique `.env` files per environment (`.env.production`, `.env.staging`, etc.) and keep them outside version control.

## 🧰 Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Defaults ignoring `.env` | File not in repo root or wrong encoding | Ensure `.env` is UTF-8 in project root; rerun `config.py` |
| `DATA_PATH not found` | Relative path from Docker container | Mount dataset into container and use absolute in-container path |
| `IMAGE_API_PROVIDER must be one of ...` | Typo in provider name | Choose `local`, `pollinations`, or `openai` only |
| Slow startups | Forced rebuild or missing cache | Keep `FORCE_REBUILD=false`, verify `index/` write permissions |
| CUDA OOM during image gen | Model too large for VRAM | Drop resolution, switch to SDXL Turbo, or fallback to Pollinations |
| Validation errors on numbers | Environment values quoted or out of range | Remove stray quotes and respect ranges (e.g. `0 ≤ TEMPERATURE ≤ 2`) |

## ✅ Summary Checklist

- [x] Copy `.env.example` ➜ `.env` and customise.
- [x] Run `uv run python config.py` until validation passes.
- [x] Execute `setup_improvements.ps1` after pulling new datasets or upgrading embeddings.
- [x] Decide on an image provider (`local` vs `pollinations` vs `openai`) and set related vars.
- [x] Export `HF_TOKEN` if using gated Stable Diffusion models.

With these guardrails, the RAG stack stays reproducible, secure, and tuned for the higher-precision retrieval pipeline introduced in the latest iteration.
