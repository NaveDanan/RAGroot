# System Architecture

Comprehensive overview of the 2025 release of the GenAI RAG application, including the latest retrieval, LaTeX, and image-generation upgrades.

## 🏗️ High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           Docker Image                           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 FastAPI Application (main.py)               │  │
│  │  • Lifecycle + dependency wiring                            │  │
│  │  • Health, stats, /answer, /stream endpoints                │  │
│  │  • Global config + hash-based index caching                 │  │
│  └──────────────┬──────────────────────────────────────────────┘  │
│                 │                                                 │
│  ┌──────────────▼──────────────┐      ┌────────────────────────┐  │
│  │  Configuration Layer        │      │  Evaluation & Tooling  │  │
│  │  (config.py, .env)          │      │  (evaluate_rag.py,     │  │
│  │  • Validation               │      │   setup_* scripts)      │  │
│  │  • Secrets masking          │      │  • Quality scoring      │  │
│  │  • Runtime overrides        │      │  • Regression checks    │  │
│  └──────────────┬──────────────┘      └───────────┬────────────┘  │
│                 │                                 │               │
│  ┌──────────────▼──────────────┐      ┌───────────▼────────────┐  │
│  │   Vector Indexer            │◄────►│   RAG Pipeline         │  │
│  │   (indexer.py)              │      │   (retriever.py)        │  │
│  │   • JSONL streaming         │      │   • LaTeX-aware prompts │  │
│  │   • all-mpnet-base-v2       │      │   • Phi-3 Mini (GGUF)   │  │
│  │     embeddings (768-dim)    │      │   • Strict anti-halluc. │  │
│  │   • FAISS + semantic rerank │      │   • SSE streaming       │  │
│  │   • LaTeX preprocessing     │      └───────────┬────────────┘  │
│  └──────────────┬──────────────┘                  │               │
│                 │                                 │               │
│  ┌──────────────▼──────────────┐      ┌───────────▼────────────┐  │
│  │   LaTeX Processing          │      │   Image Generation     │  │
│  │   (latex_utils.py)          │      │   (image_gen.py)       │  │
│  │   • Inline/display parsing  │      │   • Provider strategy   │  │
│  │   • Symbol expansion        │      │   • Pollinations (free) │  │
│  │   • Dual track (search/UI)  │      │   • Local SD 3.5 /      │  │
│  │                             │      │     SDXL / Qwen         │  │
│  │                             │      │     SDXL / Qwen         │  │
│  └──────────────┬──────────────┘      │   • Async base64 URLs   │  │
│                 │                     └───────────┬────────────┘  │
│  ┌──────────────▼──────────────┐                  │               │
│  │ Persistence & Assets        │                  │               │
│  │  • index/ (FAISS + hash)    │                  │               │
│  │  • models/ (GGUF + SD)      │                  │               │
│  │  • static/ (UI, images)     │◄─────────────────┘               │
│  └────────────────────────────────────────────────────────────────┘
│                                                                  │
│                   Web UI (static/index.html)                     │
│                   • MathJax rendering                            │
│                   • SSE stream listener                          │
│                   • Image previews + stats                       │
└──────────────────────────────────────────────────────────────────┘
```

## 📋 Component Deep Dive

### FastAPI Service (`main.py`)
- Boots configuration, indexer, and retrieval pipeline once per container (singleton pattern).
- Exposes `/answer`, `/stream`, `/health`, `/stats`, and `/image/providers` endpoints with structured JSON.
- Performs dataset hash checks on startup to reuse cached FAISS indices, falling back to a full rebuild when source data changes.
- Surfaces runtime diagnostics (documents indexed, embedding dimension, image provider readiness) through the statistics API and logs.

### Configuration Layer (`config.py` + `.env` + setup scripts)
- Central authority for every tunable parameter with Pydantic validation, range checks, and path verification.
- Masks sensitive values (API keys, HF tokens) when printing or logging configuration.
- Ships automation helpers: `setup.ps1`, `setup_improvements.ps1`, and `setup_local_image_gen.ps1` for one-command environment preparation.
- Key defaults after the 2025 improvements:
  - `EMBEDDING_MODEL=all-mpnet-base-v2`
  - `IMAGE_API_PROVIDER=local` (fallbacks: `pollinations`, `openai`)
  - `IMAGE_MODEL_NAME` & `IMAGE_MODEL_PATH` support Stable Diffusion 3.5, SDXL Turbo, SDXL Base, or Qwen Lightning.

### Vector Indexer (`indexer.py`)
- Streams `.jsonl` input to avoid peak memory spikes; validates every record before ingestion.
- Uses `all-mpnet-base-v2` (768-dim) embeddings for 15–20% richer semantics compared to the legacy MiniLM model.
- Persists FAISS IndexFlatL2 indices, document metadata, and dataset hash to `index/` for instant warm starts.
- Implements semantic reranking: retrieve `top_k * RERANK_MULTIPLIER`, re-score with cosine similarity on “title + abstract”, then return the highest quality results.
- Integrates LaTeX preprocessing so expressions like `$O(n^2)$` become searchable (`O(n squared)`), while the untouched raw text remains available for presentation.

### LaTeX Engine (`latex_utils.py`)
- Detects inline `$...$` and block `$$...$$` segments, expands 60+ mathematical symbols, and rewrites superscripts/subscripts into natural language for embedding and LLM consumption.
- Maintains a dual-track representation: processed text for retrieval quality, original LaTeX for UI rendering via MathJax.
- Shared by the indexer (during ingestion and query encoding) and retriever (when crafting the LLM prompt).

### Retrieval & Generation (`retriever.py`)
- Tokenizes incoming queries, obtains embeddings, calls the indexer, and builds deterministic prompts.
- Enforces anti-hallucination guardrails: citations mandatory, no mixing of documents, structured answer template (Topic → Approach → Technical Details → Results).
- Runs Phi-3 Mini 4K Instruct (Q4 quantized) through `llama.cpp` bindings with tuned generation parameters (`temperature=0.3`, `top_p=0.85`, `max_tokens=600`, `repeat_penalty=1.1`).
- Supports streaming via Server-Sent Events, forwarding partial generations to the UI for better perceived latency.

### Image Generation (`image_gen.py`)
- Strategy pattern that selects between:
  1. **Local providers** (Stable Diffusion 3.5 Medium, SDXL Turbo/Base, Qwen-Image Lightning) with lazy model download, CUDA/CPU detection, and aggressive VRAM optimisations.
  2. **Pollinations** (zero-config, free) for instant demos.
  3. **OpenAI DALL·E** for premium quality with API-key gating.
- Returns base64 data URLs stored temporarily under `static/generated_images/` for seamless embedding in responses and the web UI.
- Includes scripts (`setup_local_image_gen.ps1`, `tools/test_image_gen.py`, `SD35_SETUP_GUIDE.md`) to validate hardware, authenticate gated models, and benchmark throughput.

### Web Experience (`static/index.html`)
- Pure HTML/CSS/JS bundle with no build process; compatible with Docker’s static file serving.
- Renders math notation via MathJax, displays streaming responses token-by-token, and toggles image generation requests.
- Pulls system statistics on load, showing document counts, embedding dimensions, and provider status.

### Quality Gateways & Tooling
- `evaluate_rag.py` benchmarks keyword coverage, citation accuracy, specificity, and length—targeting 78/100 (Grade B) after improvements.
- Test scripts (`test_api.py`, `tools/test_image_gen.py`) provide regression checks for both the text and image pipelines.
- Documentation established for every subsystem (architecture, configuration, LaTeX, image setup, improvement log).

## 🔄 Data & Request Flow

### Indexing Pipeline

1. **Dataset ingestion** – stream JSONL line-by-line to keep RAM usage predictable.
2. **Validation** – confirm required keys (`id`, `title`, `abstract`) and skip malformed rows.
3. **LaTeX processing** – expand symbols for embedding, preserve originals for later display.
4. **Batch embedding** – encode abstracts with `all-mpnet-base-v2` in configurable batch sizes.
5. **FAISS build + cache** – construct IndexFlatL2, persist vectors, document metadata, and dataset hash to `index/`.

Typical performance (2.9k docs): ~3–4 minutes CPU indexing, peak memory <2.5 GB.

### Query & Generation Pipeline

1. **Frontend submission** – UI or API client posts to `/answer` with optional `generate_image` flag.
2. **Configuration check** – resolves runtime parameters (threads, top_k, image provider).
3. **Query preprocessing** – run through LaTeX handler, embed via `all-mpnet-base-v2`.
4. **Semantic retrieval + rerank** – gather `top_k * multiplier`, rescore, keep highest quality.
5. **Prompt assembly** – apply strict template, include citations, highlight constraints.
6. **LLM inference** – Phi-3 Mini generates answer; SSE channel streams tokens if requested.
7. **Optional image generation** – convert answer into visual prompt, call selected provider, embed base64 image URL.
8. **Response packaging** – return answer, citations, context passages, latency metrics, and image metadata.

Observed latencies (CPU-only server):
- Embedding + retrieval + rerank: ~250 ms
- LLM completion: 5–15 s (temperature 0.3)
- Image generation: 5–12 s (GPU local) to 16 min (Qwen Lightning with CPU offload) depending on provider.

## ⚙️ Architectural Patterns & Trade-offs

- **Singletons** – long-lived indexer, retriever, and image generator instances prevent redundant model loads inside the container.
- **Strategy** – image providers and future retrieval policies can be swapped without touching caller logic.
- **Builder** – prompt construction assembles role, instructions, context, and user query deterministically (critical for reproducible evaluations).
- **Hash-based caching** – dataset SHA-256 fingerprinting avoids costly reindex operations when content is unchanged.
- **Dual-track text representation** – LaTeX support differentiates between searchable and display-ready forms of the same document.

### Performance Considerations
- Upgrading to `all-mpnet-base-v2` increases embedding time by ~20 ms per batch but improves retrieval precision by 15–20%.
- Semantic reranking adds ~150 ms per query yet removes most “wrong top document” cases observed in earlier evaluations.
- Lowering `temperature` to 0.3 and nudging `top_p` to 0.85 prioritises factual accuracy over stylistic variety.
- Local image generation delivers offline capability but demands 10–15 GB disk per model and ≥12 GB VRAM for best results; Pollinations remains a fast fallback.

## 🔐 Security & Reliability
- Environment-driven configuration prevents secrets from being hardcoded and keeps `.env` files out of version control.
- Request schemas constrain `top_k`, timeouts, and token budgets to guard against abuse.
- Optional Hugging Face authentication (`HF_TOKEN`) enables gated model downloads while avoiding token leakage in logs.
- Docker volume mounts expose datasets read-only and isolate generated assets inside the container.

## 📈 Scalability Outlook
- Current single-process design comfortably serves <100 concurrent requests; scaling paths include Gunicorn workers, horizontal pod autoscaling, or vector-database offloading (pgvector/Qdrant).
- For corpora >100k documents, consider sharded FAISS indexes or migrating to a managed vector store.
- Heavy image workloads can be separated into an async queue (RQ/Celery) or externalised to cloud inference endpoints.

## 🧪 Quality Assurance
- `evaluate_rag.py` provides deterministic scoring across four metrics. The improvements playbook (`IMPROVEMENTS.md`, `IMPROVEMENTS_SUMMARY.md`) captures the before/after deltas (50 ➜ 78 overall score).
- Test harnesses (`test_api.py`, `tools/test_image_gen.py`) cover REST correctness and GPU readiness.
- Documentation underpins onboarding: architecture, configuration, LaTeX, local image generation, and Stable Diffusion setup guides.

## 🔮 Roadmap Ideas
- Hybrid retrieval (BM25 + dense), cross-encoder reranking, and query expansion for even higher recall.
- Enhanced Phi-3 prompts with few-shot exemplars or migration to >8k-context models when hardware permits.
- UI polish: conversational chat mode, query history, export/share flows, and MathJax-enabled PDF export.
- Analytics: capture usage metrics, feedback loops, and automated evaluation dashboards.

## ✅ Key Takeaways
- Swapping to `all-mpnet-base-v2`, enforcing strict prompts, and adding reranking eliminated the hallucinations and vague answers identified in earlier reviews.
- LaTeX support now spans ingestion → retrieval → UI, making the system research-paper ready.
- Image generation is pluggable: free Pollinations for demos, local Stable Diffusion/Qwen for offline control, or OpenAI for premium output.
- The configuration layer, tooling scripts, and documentation collectively make the system reproducible, auditable, and production friendly.

This architecture balances quality, performance, and maintainability while remaining deployable with a single Docker command.
