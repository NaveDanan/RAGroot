# GenAI RAG Application

A fully self-contained Retrieval-Augmented Generation (RAG) system for querying academic abstracts using local LLMs and vector search.

## 🚀 Features

- **Fully On-Premise**: All core components run locally inside Docker
- **Offline Operation**: All embedding models can be stored locally for air-gapped deployment
- **Vector Search**: FAISS-based semantic search with sentence-transformers
- **Local LLM**: Llama-3.2-3B-Instruct-Q4_K_M for efficient CPU inference
- **Web UI**: Clean, modern interface for querying and viewing results
- **REST API**: `/answer` and `/stream` endpoints for programmatic access
- **Smart Indexing**: Automatic detection of dataset changes with hash-based caching
- **Bonus: Image Generation**: Optional integration with Pollinations.ai (free) or OpenAI DALL-E

## 📋 Requirements

- **Docker** (with at least 8GB RAM allocated)
- **CPU**: Works on CPU (4+ cores recommended)
- **GPU**: Optional, for faster inference
- **Disk Space**: ~5GB for Docker image (includes models)

## 🏗️ Architecture

```
┌─────────────┐
│   Dataset   │ (.jsonl)
│ (abstracts) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Indexer    │ (sentence-transformers + FAISS)
│  Embeddings │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│  Retriever  │ ───▶ │   Local LLM  │ (Llama-3.2-3B)
│  (RAG)      │      │  (llama.cpp) │
└──────┬──────┘      └──────────────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Web UI    │      │  Image Gen   │ (Optional)
│   (FastAPI) │ ───▶ │ (External)   │
└─────────────┘      └──────────────┘
```

## 🛠️ Quick Start

### 1. Build the Docker Image

```bash
# Clone or create the project directory
mkdir genai-rag && cd genai-rag

# Add all the code files (main.py, indexer.py, retriever.py, image_gen.py, index.html)

# Build the image
docker build -t navedanan/genai-app:latest .
```

### 2. Run the Application

```bash
docker run --rm -p 8080:8080 \
  -e DATA_PATH=/data/arxiv_2.9k.jsonl \
  -v $(pwd)/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro \
  navedanan/genai-app:latest
```

### 3. Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:8080
```
### 4. Jsonl Structure Example
```bash
{
    "id": "2509.21245v1", 
    "submitter": "Team Hunyuan3D", 
    "authors": "Team Hunyuan3D, :, Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jingwei Huang, Junlin Yu, Kunhong Li, Linus, Penghao Wang, Qingxiang Lin, Sicong Liu, Xianghui Yang, Yixuan Tang, Yunfei Zhao, Zeqiang Lai, Zhihao Liang, Zibo Zhao", 
    "title": "Hunyuan3D-Omni: A Unified Framework for Controllable Generation of 3D Assets", 
    "comments": "Technical Report; 3D Generation", 
    "journal-ref": "", 
    "doi": "", 
    "categories": "cs.CV cs.AI", 
    "abstract": "Recent advances in 3D-native generative models have accelerated asset creation for games, film, and design. However, most methods still rely primarily on image or text conditioning and lack fine-grained, cross-modal controls, which limits controllability and practical adoption. To address this gap, we present Hunyuan3D-Omni, a unified framework for fine-grained, controllable 3D asset generation built on Hunyuan3D 2.1. In addition to images, Hunyuan3D-Omni accepts point clouds, voxels, bounding boxes, and skeletal pose priors as conditioning signals, enabling precise control over geometry, topology, and pose. Instead of separate heads for each modality, our model unifies all signals in a single cross-modal architecture. We train with a progressive, difficulty-aware sampling strategy that selects one control modality per example and biases sampling toward harder signals (e.g., skeletal pose) while downweighting easier ones (e.g., point clouds), encouraging robust multi-modal fusion and graceful handling of missing inputs. Experiments show that these additional controls improve generation accuracy, enable geometry-aware transformations, and increase robustness for production workflows.", 
    "source": "arxiv"
}
```
## 📁 Project Structure

```
genai-rag/
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── main.py                # FastAPI application
├── indexer.py             # Vector indexing (FAISS)
├── retriever.py           # RAG pipeline with LLM
├── image_gen.py           # Image generation (bonus)
└── index.html             # Web UI
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `/data/arxiv_2.9k.jsonl` | Path to dataset file |
| `INDEX_DIR` | `/app/index` | Directory for vector index |
| `MODEL_PATH` | `/app/models/llama-model.gguf` | Path to LLM model |
| `IMAGE_API_PROVIDER` | `pollinations` | Image generation provider |
| `IMAGE_API_KEY` | (empty) | API key for OpenAI (if using) |

### Image Generation Setup (Optional)

#### Option 1: Pollinations.ai (Default, Free)
No configuration needed! It works out of the box.

```bash
docker run --rm -p 8080:8080 \
  -e DATA_PATH=/data/arxiv_2.9k.jsonl \
  -v $(pwd)/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro \
  navedanan/genai-app:latest
```

#### Option 2: OpenAI DALL-E
Requires an OpenAI API key:

```bash
docker run --rm -p 8080:8080 \
  -e DATA_PATH=/data/arxiv_2.9k.jsonl \
  -e IMAGE_API_PROVIDER=openai \
  -e IMAGE_API_KEY=sk-your-api-key-here \
  -v $(pwd)/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro \
  navedanan/genai-app:latest
```

## 📡 API Endpoints

### POST `/answer`
Query the system and get a complete answer.

**Request:**
```json
{
  "query": "What are recent advances in transformers?",
  "generate_image": false,
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Recent advances in transformers include...",
  "citations": [
    {"doc_id": "2509.01234", "title": "A New Approach to Transformers"}
  ],
  "retrieved_context": ["Abstract text..."],
  "image_url": null
}
```

### POST `/stream`
Stream the answer generation (Server-Sent Events).

```bash
curl -X POST http://localhost:8080/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about neural networks"}'
```

### GET `/health`
Check system health and statistics.

```bash
curl http://localhost:8080/health
```

### GET `/stats`
Get indexing statistics.

```bash
curl http://localhost:8080/stats
```

## 🎯 Dataset Format

The system expects a `.jsonl` file where each line is a JSON object. Based on the arxiv dataset structure:

### Complete Example Record
```json
{
  "id": "2509.21245v1",
  "submitter": "Team Hunyuan3D",
  "authors": "Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, ...",
  "title": "Hunyuan3D-Omni: A Unified Framework for Controllable Generation of 3D Assets",
  "comments": "Technical Report; 3D Generation",
  "journal-ref": "",
  "doi": "",
  "categories": "cs.CV cs.AI",
  "abstract": "Recent advances in 3D-native generative models have accelerated asset creation for games, film, and design. However, most methods still rely primarily on image or text conditioning and lack fine-grained, cross-modal controls...",
  "source": "arxiv"
}
```

### Field Requirements

**Required fields** (system will skip records without these):
- `id`: Unique document identifier (e.g., "2509.21245v1")
- `title`: Paper title (used for citations and reranking)
- `abstract`: Full abstract text (used for embeddings and retrieval)

**Optional fields** (preserved but not used):
- `authors`: Author names (comma-separated)
- `submitter`: Paper submitter
- `categories`: Subject categories (e.g., "cs.CV cs.AI")
- `comments`: Additional metadata
- `journal-ref`: Journal reference
- `doi`: Digital Object Identifier
- `source`: Data source (e.g., "arxiv")

### Validation

The system automatically:
1. ✅ Validates required fields on each record
2. ⚠️ Logs warnings for malformed records
3. ⏭️ Skips invalid records and continues processing
4. 📊 Reports total valid records indexed

### Example Validation Output
```
Processed 500 documents...
Line 234: Missing required fields, skipping
Line 567: JSON decode error, skipping - Expecting ',' delimiter
Processed 1000 documents...
...
Loaded 2897 documents (3 records skipped)
```

## 🔄 Updating the Dataset

The system automatically detects dataset changes:

1. Mount a new dataset file with a different path:
```bash
docker run --rm -p 8080:8080 \
  -e DATA_PATH=/data/new_dataset.jsonl \
  -v $(pwd)/new_dataset.jsonl:/data/new_dataset.jsonl:ro \
  navedanan/genai-app:latest
```

2. The system will:
   - Compute the file hash
   - Compare with cached hash
   - Rebuild the index if changed
   - Reuse existing index if unchanged

## 🚀 Performance Optimization

### CPU Optimization
The default configuration is optimized for CPU:
- Uses Q4 quantized Llama-3.2 (~2GB)
- 4 CPU threads for inference
- Batch embedding generation

### GPU Acceleration (Optional)
To enable GPU support:

1. Modify `Dockerfile` to use GPU-enabled llama-cpp:
```dockerfile
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

2. Run with GPU:
```bash
docker run --rm --gpus all -p 8080:8080 \
  -e DATA_PATH=/data/arxiv_2.9k.jsonl \
  -v $(pwd)/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro \
  navedanan/genai-app:latest
```

3. Modify `retriever.py` to set `n_gpu_layers`:
```python
self.llm = Llama(
    model_path=model_path,
    n_gpu_layers=35,  # Adjust based on GPU memory
    ...
)
```

## 🧪 Testing

### Test with sample query
```bash
curl -X POST http://localhost:8080/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are recent advances in natural language processing?",
    "generate_image": false,
    "top_k": 5
  }'
```

### Test image generation
```bash
curl -X POST http://localhost:8080/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain deep learning architectures",
    "generate_image": true,
    "top_k": 3
  }'
```

## 📊 Model Information

### Embedding Model
- **Model**: `all-MiniLM-L6-v2`
- **Size**: ~80MB
- **Dimension**: 384
- **Speed**: ~14,000 sentences/sec on CPU

### LLM Model
- **Model**: Llama-3.2-3B-Instruct-Q4_K_M
- **Size**: ~2.4GB
- **Context**: 4K tokens
- **Speed**: ~10-20 tokens/sec on CPU (4 cores)

## 🎨 Web UI Features

- **Query Input**: Type your question
- **Results Display**: Answer, citations, and context
- **Image Toggle**: Enable/disable image generation
- **Top-K Selection**: Choose number of documents to retrieve (3, 5, or 10)
- **Live Statistics**: View indexed document count

## 🐛 Troubleshooting

### Port already in use
```bash
# Use a different port
docker run --rm -p 8081:8080 ...
# Access at http://localhost:8081
```

### Out of memory
```bash
# Increase Docker memory allocation or use smaller model
# Edit Dockerfile to use TinyLlama instead of Phi-3
```

### Slow inference
- Reduce `top_k` parameter
- Use fewer CPU threads
- Consider GPU acceleration

### Dataset not found
```bash
# Verify the volume mount path matches DATA_PATH
docker run --rm -p 8080:8080 \
  -e DATA_PATH=/data/arxiv_2.9k.jsonl \
  -v /absolute/path/to/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro \
  navedanan/genai-app:latest
```

## � Documentation

For detailed setup and configuration guides, see:

- **[Offline/On-Premise Setup](Documentation/OFFLINE_SETUP.md)** - Complete guide for air-gapped deployment
- **[Configuration Guide](Documentation/CONFIGURATION.md)** - All configuration options explained
- **[Architecture Overview](Documentation/ARCHITECTURE.md)** - System design and components
- **[Quick Start Guide](Documentation/QUICKSTART.md)** - Get up and running in minutes

### Offline Operation

To run completely offline with all models stored locally:

```bash
# 1. Download all models (requires internet, one-time)
python tools/download_models.py --all

# 2. Configure for offline mode (edit .env)
EMBEDDING_LOCAL_ONLY=true
EMBEDDING_CACHE_DIR=models/embeddings

# 3. Validate offline setup
python tools/validate_offline.py

# 4. Run the application
python main.py
```

See [Documentation/OFFLINE_SETUP.md](Documentation/OFFLINE_SETUP.md) for detailed instructions.

## �📝 License

This project is provided as-is for the GenAI Engineer assignment.

## 🙏 Acknowledgments

- **sentence-transformers**: Semantic embeddings
- **FAISS**: Efficient vector search
- **llama.cpp**: Efficient LLM inference
- **Phi-3**: Microsoft's compact language model
- **Pollinations.ai**: Free image generation
- **FastAPI**: Modern web framework

## 📧 Support

For issues or questions, please refer to the documentation or create an issue in the repository.
