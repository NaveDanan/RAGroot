# CLI Command Guide

## Overview

The GenAI RAG Application provides a comprehensive command-line interface for indexing, querying, serving, and evaluating the system without using the web UI.

## Usage

```bash
python main.py <command> [options]
```

## Available Commands

### 1. `index` - Build Vector Index

Build a vector index from a JSONL dataset.

**Syntax:**
```bash
python main.py index --data <dataset-path> [--output <index-directory>]
```

**Arguments:**
- `--data` (required): Path to JSONL dataset file
- `--output` (optional): Output directory for index (default: `index/`)

**Example:**
```bash
python main.py index --data data/arxiv_2.9k.jsonl --output index/
```

**Output:**
```
══════════════════════════════════════════════════════════════════════
📦 Building Vector Index
══════════════════════════════════════════════════════════════════════
Data source:      data/arxiv_2.9k.jsonl
Output directory: index/
══════════════════════════════════════════════════════════════════════

Loading embedding model...
Embedding model loaded (dim=768)
Reading dataset from data/arxiv_2.9k.jsonl
Processed 500 documents...
Processed 1000 documents...
...
Loaded 2900 documents
Generating embeddings...
Encoded 500 documents...
...
Generated embeddings: (2900, 768)
Building FAISS index...
FAISS index built with 2900 vectors
Saving index to disk...
Index saved successfully

══════════════════════════════════════════════════════════════════════
✅ Index Built Successfully!
══════════════════════════════════════════════════════════════════════
Total documents: 2900
Index size: {'vectors': 2900, 'dimension': 768}
══════════════════════════════════════════════════════════════════════
```

---

### 2. `query` - Query the System

Query the RAG system directly from the command line.

**Syntax:**
```bash
python main.py query "<query-text>" [--top-k <number>] [--index-dir <path>] [--model <path>]
```

**Arguments:**
- `query` (required): Query text (use quotes for multi-word queries)
- `--top-k` (optional): Number of documents to retrieve (default: 5)
- `--index-dir` (optional): Index directory path (default: `index/`)
- `--model` (optional): LLM model path (default: `models/llama-model.gguf`)

**Examples:**
```bash
# Simple query
python main.py query "What are recent advances in transformers?"

# Query with custom top-k
python main.py query "Explain federated learning" --top-k 10

# Query with custom paths
python main.py query "How does GRPO work?" --index-dir custom_index/ --model models/custom.gguf
```

**Output:**
```
══════════════════════════════════════════════════════════════════════
🔍 Querying RAG System
══════════════════════════════════════════════════════════════════════
Query: What are recent advances in transformers?
Top-K: 5
══════════════════════════════════════════════════════════════════════

Loading index from disk...
Index loaded: 2900 documents
Loading LLM from models/llama-model.gguf
LLM loaded successfully
Answering query: What are recent advances in transformers?
Generating answer with LLM...
Answer generated successfully

📝 Answer:
──────────────────────────────────────────────────────────────────────
Recent advances in transformers include improved attention mechanisms,
efficient training methods, and novel architectures for specific tasks...
[Full answer displayed here]
──────────────────────────────────────────────────────────────────────

📚 Citations:
  [1] 2509.01234: A New Approach to Transformers
  [2] 2509.01567: Efficient Attention Mechanisms
  [3] 2509.02341: Scaling Transformers to Billions of Parameters
  ...

📄 Retrieved Context (first 200 chars each):
  [1] We propose a novel attention mechanism that reduces complexity...
  [2] This paper introduces an efficient method for training large...
  ...
```

---

### 3. `serve` - Start Web Server

Start the FastAPI web server for browser-based interaction.

**Syntax:**
```bash
python main.py serve [--host <address>] [--port <number>] [--reload]
```

**Arguments:**
- `--host` (optional): Host address to bind (default: `0.0.0.0`)
- `--port` (optional): Port number (default: `8080`)
- `--reload` (optional): Enable auto-reload for development

**Examples:**
```bash
# Start server on default port 8080
python main.py serve

# Start on custom port
python main.py serve --port 9000

# Start with auto-reload for development
python main.py serve --reload

# Start on localhost only
python main.py serve --host 127.0.0.1 --port 8080
```

**Output:**
```
══════════════════════════════════════════════════════════════════════
🚀 Starting GenAI RAG Server
══════════════════════════════════════════════════════════════════════
Host: 0.0.0.0
Port: 8080
URL:  http://localhost:8080
══════════════════════════════════════════════════════════════════════

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Initializing system with DATA_PATH: data/arxiv_2.9k.jsonl
INFO:     Dataset unchanged, loading existing index
INFO:     Loading index from disk...
INFO:     Index loaded: 2900 documents
INFO:     Loading LLM from models/llama-model.gguf
INFO:     LLM loaded successfully
INFO:     System initialized successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**Access:**
- Web UI: http://localhost:8080
- API Docs: http://localhost:8080/docs
- Health: http://localhost:8080/health
- Stats: http://localhost:8080/stats

---

### 4. `evaluate` - Run Evaluation

Run the evaluation test suite to measure system quality.

**Syntax:**
```bash
python main.py evaluate [--url <api-url>]
```

**Arguments:**
- `--url` (optional): API base URL (default: `http://localhost:8080`)

**Examples:**
```bash
# Evaluate local server
python main.py evaluate

# Evaluate remote server
python main.py evaluate --url http://example.com:8080
```

**Output:**
```
══════════════════════════════════════════════════════════════════════
📊 Running Evaluation
══════════════════════════════════════════════════════════════════════
API URL: http://localhost:8080
══════════════════════════════════════════════════════════════════════

RAG SYSTEM EVALUATION
══════════════════════════════════════════════════════════════════════
✓ System is healthy
  - Indexed documents: 2900

RUNNING TEST CASES
══════════════════════════════════════════════════════════════════════

[Test 1/4] Category: specific_paper
Query: What is the paper 'Explaining Fine Tuned LLMs...' about?

  Response time: 8.3s
  Answer length: 145 words
  Citations: 5

  SCORES:
    Keyword Coverage:    32.0/40
    Citation Relevance:  30.0/30
    Specificity:         16.0/20
    Length:              10.0/10
    TOTAL:               88.0/100

  Top Citation: [2509.21241v1] Explaining Fine Tuned LLMs...
  
  Answer: The paper introduces CFFTLLMExplainer, a counterfactual...

[Test 2/4] Category: technical_method
...

OVERALL RESULTS
══════════════════════════════════════════════════════════════════════

Average Scores:
  Keyword Coverage:    28.5/40
  Citation Relevance:  25.0/30
  Specificity:         15.5/20
  Length:              9.0/10
  OVERALL SCORE:       78.0/100

  GRADE: B (Good)

Recommendations:
  - Keyword coverage is good but can be improved
  - Citation relevance is excellent
  - Specificity is strong with technical details
```

---

## Quick Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `index` | Build vector index | `python main.py index --data data.jsonl` |
| `query` | Ask a question | `python main.py query "What is AI?"` |
| `serve` | Start web server | `python main.py serve --port 8080` |
| `evaluate` | Run tests | `python main.py evaluate` |

---

## Environment Variables

All CLI commands respect environment variables from `.env`:

- `DATA_PATH` - Default dataset path
- `INDEX_DIR` - Default index directory
- `MODEL_PATH` - Default LLM model path
- `PORT` - Default server port
- `HOST` - Default server host

---

## Tips & Best Practices

### 1. First-Time Setup
```bash
# 1. Index your dataset first
python main.py index --data data/arxiv_2.9k.jsonl

# 2. Test with a query
python main.py query "What is deep learning?"

# 3. Start the server
python main.py serve
```

### 2. Development Workflow
```bash
# Use --reload for development
python main.py serve --reload

# Test queries quickly from CLI
python main.py query "test query" --top-k 3
```

### 3. Production Deployment
```bash
# Use specific port and host
python main.py serve --host 0.0.0.0 --port 8080

# Or use Docker for production
docker run -p 8080:8080 navedanan/genai-app:latest
```

### 4. Testing & Validation
```bash
# Evaluate system quality
python main.py evaluate

# Test with different top-k values
python main.py query "test" --top-k 3
python main.py query "test" --top-k 10
```

---

## Troubleshooting

### Command not recognized
```bash
# Make sure you're in the project directory
cd /path/to/RAG_PDF

# Activate virtual environment if using one
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Index not found
```bash
# Build index first
python main.py index --data data/arxiv_2.9k.jsonl

# Or specify index directory
python main.py query "test" --index-dir /path/to/index
```

### Model not found
```bash
# Check model path in .env or specify directly
python main.py query "test" --model /path/to/model.gguf
```

### Port already in use
```bash
# Use different port
python main.py serve --port 8081
```

---

## Advanced Usage

### Batch Queries
Create a script to run multiple queries:

```bash
#!/bin/bash
queries=(
    "What is machine learning?"
    "Explain neural networks"
    "How does backpropagation work?"
)

for query in "${queries[@]}"; do
    echo "Query: $query"
    python main.py query "$query" --top-k 3
    echo "---"
done
```

### Custom Evaluation
```bash
# Run evaluation with custom test cases
# Edit tests/evaluate_rag.py and run
python main.py evaluate --url http://localhost:8080
```

### Performance Testing
```bash
# Time a query
time python main.py query "What is AI?" --top-k 5

# Monitor server logs
python main.py serve --port 8080 2>&1 | tee server.log
```

---

## Integration with Other Tools

### Using with curl
```bash
# Start server
python main.py serve &

# Query via API
curl -X POST http://localhost:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "top_k": 5}'
```

### Using with Python scripts
```python
import subprocess
import json

# Run query and capture output
result = subprocess.run(
    ["python", "main.py", "query", "What is AI?"],
    capture_output=True,
    text=True
)
print(result.stdout)
```

---

## Getting Help

```bash
# Show all commands
python main.py --help

# Show help for specific command
python main.py index --help
python main.py query --help
python main.py serve --help
python main.py evaluate --help
```

---

**Version:** 1.0  
**Last Updated:** October 6, 2025
