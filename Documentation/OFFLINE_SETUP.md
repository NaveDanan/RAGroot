# Offline / On-Premise Setup Guide

This guide explains how to set up the RAG system for **completely offline operation** with all models stored locally.

## Overview

For on-premise deployment without internet access, all required models must be downloaded and cached locally:

1. **Embedding Models** - For document and query encoding
2. **LLM Model** - For answer generation (GGUF format)
3. **Image Generation Models** (optional) - For visual outputs

## Prerequisites

- Python 3.8+
- Sufficient disk space (varies by models, typically 5-20 GB)
- Internet connection for initial model download
- All required Python packages installed

## Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt
```

## Step 2: Configure for Offline Operation

Edit your `.env` file to enable local-only mode:

```bash
# Copy example configuration
cp .env.example .env

# Edit .env and set these values:
EMBEDDING_LOCAL_ONLY=true
EMBEDDING_CACHE_DIR=models/embeddings
SKIP_CHECK_ST_UPDATES=true

# Optional: Force CPU mode if no GPU available
FORCE_CPU=false
```

### Key Configuration Options

| Variable | Description | Recommended Value |
|----------|-------------|-------------------|
| `EMBEDDING_LOCAL_ONLY` | Prevents downloading models from internet | `true` |
| `EMBEDDING_CACHE_DIR` | Local directory for embedding models | `models/embeddings` |
| `EMBEDDING_MODEL` | Which embedding model to use | `all-mpnet-base-v2` |
| `SKIP_CHECK_ST_UPDATES` | Skip checking for model updates | `true` |
| `FORCE_CPU` | Force CPU-only operation | `false` (set to `true` if no GPU) |

## Step 3: Download Models

### Option A: Automatic Download (Recommended)

Use the provided download script:

```bash
# Download all models (embedding + image generation)
python tools/download_models.py --all

# Download only embedding models
python tools/download_models.py --embedding-only

# Download only image generation models
python tools/download_models.py --image-only
```

The script will:
- ✅ Download models from HuggingFace Hub
- ✅ Cache them in the specified directories
- ✅ Verify successful downloads
- ✅ Provide clear error messages if something fails

### Option B: Manual Download

#### Embedding Models

**For Standard SentenceTransformers:**

```python
from sentence_transformers import SentenceTransformer

# Download to cache directory
model = SentenceTransformer(
    'all-mpnet-base-v2',
    cache_folder='models/embeddings'
)
```

**For SPECTER2 (Scientific Papers):**

```python
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Download base model
tokenizer = AutoTokenizer.from_pretrained(
    'allenai/specter2_base',
    cache_dir='models/embeddings'
)
model = AutoAdapterModel.from_pretrained(
    'allenai/specter2_base',
    cache_dir='models/embeddings'
)

# Download adapters
model.load_adapter('allenai/specter2', source='hf')
model.load_adapter('allenai/specter2_adhoc_query', source='hf')
```

#### Image Generation Models

```python
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/sdxl-turbo',
    torch_dtype=torch.float16,
    variant='fp16',
    cache_dir='models/sdxl-turbo'
)
```

## Step 4: Verify Offline Setup

After downloading models, verify they work offline:

```bash
# Set environment to force offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Or on Windows PowerShell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"

# Run the application
python main.py
```

The application should start without attempting to download anything.

## Step 5: Build the Index

```bash
# Build vector index from your dataset
python -m utils.indexer
```

This creates:
- `index/faiss.index` - Vector search index
- `index/documents.pkl` - Document metadata
- `index/embeddings.npy` - Cached embeddings

## Step 6: Run the Application

```bash
# Start the server
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8080
```

Access the web interface at: `http://localhost:8080`

## Supported Embedding Models

### Standard Models (via sentence-transformers)

All stored in `EMBEDDING_CACHE_DIR`:

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | ~80 MB | ⚡⚡⚡ Fast | Good | General use, fast inference |
| `all-mpnet-base-v2` | ~420 MB | ⚡⚡ Medium | Better | Balanced quality/speed |
| `all-MiniLM-L12-v2` | ~120 MB | ⚡⚡ Medium | Good | Middle ground |

### Scientific Paper Models (SPECTER2)

Specialized for academic papers:

| Model | Adapter | Best For |
|-------|---------|----------|
| `allenai/specter2_base` | `allenai/specter2` | Document encoding |
| `allenai/specter2_base` | `allenai/specter2_adhoc_query` | Query encoding |

## Directory Structure

After setup, your directory should look like:

```
RAG_PDF/
├── models/
│   ├── embeddings/           # Embedding model cache
│   │   ├── sentence-transformers_all-mpnet-base-v2/
│   │   └── ...
│   ├── sdxl-turbo/           # Image generation cache
│   │   └── models--stabilityai--sdxl-turbo/
│   └── llama-model.gguf      # LLM model
├── index/                     # Vector index
│   ├── faiss.index
│   ├── documents.pkl
│   └── embeddings.npy
├── data/
│   └── arxiv_2.9k.jsonl
└── .env                       # Configuration
```

## Disk Space Requirements

| Component | Typical Size |
|-----------|-------------|
| Embedding Model (MiniLM) | ~80 MB |
| Embedding Model (MPNet) | ~420 MB |
| SPECTER2 Models | ~1.5 GB |
| LLM Model (3B params) | ~2-4 GB |
| Image Model (SDXL-Turbo) | ~7 GB |
| Vector Index | ~50-500 MB (depends on dataset) |
| **Total (all components)** | ~10-15 GB |

## Troubleshooting

### Error: "Model not found in local cache"

**Solution:** Run the download script or set `EMBEDDING_LOCAL_ONLY=false` temporarily:

```bash
python tools/download_models.py --embedding-only
```

### Error: "No internet connection"

**Solution:** This is expected in offline mode. Ensure all models are pre-downloaded.

### Error: "CUDA out of memory"

**Solution:** Switch to CPU mode:

```bash
# In .env
FORCE_CPU=true
N_GPU_LAYERS=0
```

### Slow Startup

**Solution:** Models are being loaded. This is normal on first run. Subsequent runs will be faster.

### Missing Dependencies

```bash
# For SPECTER2
pip install adapter-transformers

# For image generation
pip install diffusers accelerate

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Docker Deployment

For containerized offline deployment:

```dockerfile
# Dockerfile with embedded models
FROM python:3.11-slim

# Copy models into container
COPY models/ /app/models/
COPY index/ /app/index/

# Set offline mode
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV EMBEDDING_LOCAL_ONLY=true

# Copy application
COPY . /app/
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Run application
CMD ["python", "main.py"]
```

Build and deploy:

```bash
# Build image with models
docker build -t rag-offline .

# Run container
docker run -p 8080:8080 rag-offline
```

## Air-Gapped Deployment

For completely isolated systems:

1. **On a machine with internet:**
   ```bash
   # Download all models
   python tools/download_models.py --all
   
   # Create archive
   tar -czf rag-models.tar.gz models/ index/
   ```

2. **Transfer archive to air-gapped system**

3. **On air-gapped system:**
   ```bash
   # Extract models
   tar -xzf rag-models.tar.gz
   
   # Set environment
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   
   # Run application
   python main.py
   ```

## Performance Optimization

### For CPU-Only Systems

```bash
# .env settings
FORCE_CPU=true
N_GPU_LAYERS=0
N_THREADS=8  # Match your CPU cores
EMBEDDING_BATCH_SIZE=16  # Smaller batches for CPU
```

### For GPU Systems

```bash
# .env settings
FORCE_CPU=false
N_GPU_LAYERS=35  # Offload LLM layers to GPU
EMBEDDING_BATCH_SIZE=64  # Larger batches for GPU
```

## Validation Script

Verify your offline setup:

```python
# validate_offline.py
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from utils.config import config
from utils.indexer import VectorIndexer

print("Testing offline operation...")

# Test embedding model
print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
indexer = VectorIndexer(index_dir=config.INDEX_DIR)
print("✅ Embedding model loaded successfully")

# Test search
if indexer.index is None:
    indexer.load_index()
results = indexer.search("machine learning", top_k=3)
print(f"✅ Search successful: {len(results)} results")

print("\n🎉 Offline setup validated successfully!")
```

Run with:
```bash
python validate_offline.py
```

## Security Considerations

For on-premise deployments:

1. **Disable outbound connections** in firewall
2. **Verify no external API calls** in code
3. **Set environment variables** to force offline:
   ```bash
   HF_HUB_OFFLINE=1
   TRANSFORMERS_OFFLINE=1
   EMBEDDING_LOCAL_ONLY=true
   ```
4. **Monitor network traffic** during testing
5. **Review logs** for any download attempts

## Support

If you encounter issues with offline setup:

1. Check logs in console output
2. Verify disk space: `df -h`
3. Check model cache: `ls -lh models/embeddings/`
4. Run validation script
5. Review error messages - they include helpful diagnostics

---

**Last Updated:** 2025-10-11  
**Version:** 1.0.0
