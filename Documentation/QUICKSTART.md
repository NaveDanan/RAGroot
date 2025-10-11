docker run --rm -p 8080:8080 \
docker-compose up -d
docker-compose logs -f
docker run --rm -p 8080:8080 \
docker logs -f genai-rag-app
docker stop genai-rag-app
docker restart genai-rag-app
docker run --rm -p 8081:8080 ...
docker logs genai-rag-app
# 🚀 Quick Start Guide

Spin up the latest RAG stack—complete with mpnet retrieval, LaTeX rendering, and optional local image generation—in minutes.

## ✅ Prerequisites

- **Python 3.10+** with [uv](https://github.com/astral-sh/uv) (`pip install uv`) for the local flow.
- **Docker** (optional) if you prefer containerised deployment.
- **8 GB RAM / 5 GB disk** minimum; add 10–15 GB if running Stable Diffusion locally.
- Dataset in JSONL format (default: `data/arxiv_2.9k.jsonl`).

## Path A – Local (Recommended for development)

### 1. Prepare the environment
```powershell
Copy-Item .env.example .env
./setup.ps1                 # installs deps, checks paths
./setup_improvements.ps1    # enables mpnet + reranking, rebuilds index
```

Use `./setup_local_image_gen.ps1` **only** if you plan to run Stable Diffusion / Qwen locally.

### 2. Validate configuration
```powershell
uv run python config.py
```
Ensure the checker reports “✅ Configuration is valid!”. Fix missing paths or ranges if needed.

### 3. Launch the API
```powershell
uv run python main.py
```
Wait for `INFO:main:System initialized successfully` in the console (index build takes ~3–4 minutes on first run).

### 4. Use the app
- Open `http://127.0.0.1:8000` for the MathJax-enabled UI.
- Hit `http://127.0.0.1:8000/docs` for interactive Swagger API docs.
- Stream completions from `/stream` (Server-Sent Events) to see token-by-token output.

## Path B – Docker (Portable runtime)

### 1. Build and run
```powershell
docker build -t rag-app:latest .
docker run --rm -p 8080:8080 ^
  -e DATA_PATH=/data/arxiv_2.9k.jsonl ^
  -v ${PWD}/data/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro ^
  -v ${PWD}/index:/index ^
  -v ${PWD}/models:/models ^
  rag-app:latest
```

### 2. Monitor
```powershell

```
Startup is complete when you see the same “System initialized successfully” log line.

### 3. Optional: `docker-compose up -d` to manage volumes, ports, and restarts declaratively.

## 🔍 Verify Everything

### Web UX
1. Browse to `http://localhost:8080`.
2. Submit a query (e.g. “Explain LoRA fine-tuning”).
3. Confirm: structured answer, inline LaTeX rendering, citations, retrieved context, optional image.

### Command-line regression tests
```powershell
uv run python test_api.py
```
Runs happy-path API checks using the current config.

### Quality evaluation (optional but recommended)
```powershell
python evaluate_rag.py
```
Target score: ~78/100 (Grade B) after the mpnet/reranking improvements.

## 🎨 Image Generation Options

| Provider | Setup | When to Use |
|----------|-------|------------|
| `local` *(default)* | Run `setup_local_image_gen.ps1`, set `IMAGE_MODEL_NAME` and `HF_TOKEN` if needed | Offline demos, GPU available |
| `pollinations` | Set `IMAGE_API_PROVIDER=pollinations` | No GPU, instant results, free |
| `openai` | Export `IMAGE_API_KEY` and set provider to `openai` | Highest quality via DALL·E |

Switch providers by editing `.env` and restarting `main.py` or the Docker container. Generated images are stored under `static/generated_images/`.

### Stable Diffusion 3.5 Medium (gated)
1. Request access on Hugging Face and create a read token (`HF_TOKEN`).
2. Update `.env`:
   ```env
   IMAGE_MODEL_NAME=stabilityai/stable-diffusion-3.5-medium
   IMAGE_MODEL_PATH=models/stable-diffusion-3.5-medium
   HF_TOKEN=hf_xxxxx
   ```
3. Run `setup_local_image_gen.ps1` to download and configure the model.

### Need a lighter option?
- `stabilityai/sdxl-turbo` – ~7 GB, very fast, good quality.
- `lightx2v/Qwen-Image-Lightning` – fits into 12 GB VRAM with aggressive optimisations (expect longer generation time).

## 🧠 Feature Highlights to Explore

- **LaTeX-aware retrieval** – queries like “What papers discuss $O(n^2)$ complexity?” just work.
- **Streaming API** – `/stream` returns SSE chunks for responsive UIs.
- **Statistics endpoint** – `GET /stats` shows document counts, embedding dim, active provider, and latency snapshots.
- **Reranked retrieval** – `DEFAULT_TOP_K` + `RERANK_MULTIPLIER` reduce hallucinations and enforce citation accuracy.

## 🧯 Troubleshooting Cheat Sheet

| Issue | Fix |
|-------|-----|
| Startup loops (index rebuild every time) | Ensure container or process can write to `INDEX_DIR`; keep `FORCE_REBUILD=false`. |
| `DATA_PATH not found` | Use absolute path when inside Docker (`/data/...`) and mount the dataset read-only. |
| CUDA out-of-memory on image gen | Reduce resolution, switch to SDXL Turbo, or set `IMAGE_API_PROVIDER=pollinations`. |
| UI shows raw `$...$` | Check browser console for MathJax errors; clear cache. |
| Evaluation score <70 | Verify logs show `Embedding model loaded (dim=768)` and rerun `setup_improvements.ps1` + index rebuild. |

## 📈 Performance Expectations (CPU baseline)

- **Indexing** (2.9k docs, mpnet, rerank): ~3–4 min first run, <30 s warm start.
- **Query latency**: ~250 ms retrieval + 5–15 s generation.
- **Image generation**: 5–12 s on GPU Stable Diffusion; <1 s Pollinations; ~16 min for Qwen Lightning with CPU offload.

## 🎯 Next Steps After First Run

1. Adjust `.env` to match your hardware and quality requirements.
2. Create additional benchmarks in `evaluate_rag.py` for domain-specific regressions.
3. Explore architectural docs (`ARCHITECTURE.md`, `CONFIGURATION.md`) for deeper tuning.
4. Integrate the API into your application or prototype chat UI using `/stream`.

Happy hacking! 🎉
  -e DATA_PATH=/data/arxiv_2.9k.jsonl \
