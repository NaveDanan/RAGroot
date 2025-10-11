# Docker Deployment Guide

## Overview

This guide covers building and deploying the RAG application using Docker with CUDA 12.4 support, uv package manager, and embedded models for offline operation.

## Dockerfile Features

✅ **NVIDIA CUDA 12.4** - GPU-accelerated inference  
✅ **UV Package Manager** - Fast, reliable dependency installation  
✅ **Automatic Model Download** - All models downloaded during build  
✅ **Offline Validation** - Ensures all models are present  
✅ **Production Ready** - Optimized for deployment  

## Prerequisites

### For CPU-Only Deployment
- Docker 20.10+
- 8GB+ RAM allocated to Docker
- 20GB+ free disk space

### For GPU-Accelerated Deployment
- Docker 20.10+
- NVIDIA GPU with 6GB+ VRAM
- NVIDIA Docker runtime installed
- CUDA-compatible drivers

## Installing NVIDIA Docker Runtime

### Ubuntu/Debian
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Windows with WSL2
```powershell
# Ensure WSL2 is installed and updated
wsl --update

# Install NVIDIA drivers for Windows (includes WSL2 support)
# Download from: https://www.nvidia.com/Download/index.aspx

# Test GPU access in WSL2
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Building the Docker Image

### Basic Build (Downloads Models During Build)

```bash
# Build image with default settings
docker build -t rag-app:latest .

# Build with custom tag
docker build -t myorg/rag-app:v1.0 .

# Build with build-time progress
docker build --progress=plain -t rag-app:latest .
```

**Build Time:** ~20-30 minutes (depending on internet speed and model downloads)

**Image Size:** ~15-20 GB (includes all models)

### Build with Pre-Downloaded Models

If you have models already downloaded locally to speed up build:

```bash
# Ensure models are in the correct directories
ls models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
ls models/embeddings/
ls models/sdxl-turbo/

# Build (will use local models instead of downloading)
docker build -t rag-app:latest .
```

### Multi-Architecture Build

```bash
# Build for multiple platforms (requires buildx)
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -t rag-app:latest --push .
```

## Running the Container

### CPU-Only Mode

```bash
docker run -d \
  --name rag-app \
  -p 8080:8080 \
  -v $(pwd)/data/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro \
  -v rag-index:/index \
  -e FORCE_CPU=true \
  -e N_GPU_LAYERS=0 \
  rag-app:latest
```

### GPU-Accelerated Mode (Recommended)

```bash
docker run -d \
  --name rag-app \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/data/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro \
  -v rag-index:/index \
  -e FORCE_CPU=false \
  -e N_GPU_LAYERS=35 \
  rag-app:latest
```

### With Custom Configuration

```bash
docker run -d \
  --name rag-app \
  --gpus all \
  -p 8080:8080 \
  --env-file .env.docker \
  -v $(pwd)/data:/data:ro \
  -v rag-index:/index \
  --memory="8g" \
  --cpus="4" \
  rag-app:latest
```

### Interactive Mode (for debugging)

```bash
docker run -it --rm \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/data/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro \
  rag-app:latest \
  /bin/bash
```

## Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  rag-app:
    build: .
    image: rag-app:latest
    container_name: rag-app
    ports:
      - "8080:8080"
    volumes:
      - ./data/arxiv_2.9k.jsonl:/data/arxiv_2.9k.jsonl:ro
      - rag-index:/index
    environment:
      - FORCE_CPU=false
      - N_GPU_LAYERS=35
      - EMBEDDING_LOCAL_ONLY=true
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  rag-index:
    driver: local
```

Run with Docker Compose:

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## Environment Variables

Key variables you can override:

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCE_CPU` | `false` | Force CPU-only mode |
| `N_GPU_LAYERS` | `35` | Number of LLM layers on GPU |
| `EMBEDDING_LOCAL_ONLY` | `true` | Offline mode for embeddings |
| `IMAGE_API_PROVIDER` | `pollinations` | Image generation provider |
| `DEFAULT_TOP_K` | `7` | Documents to retrieve |
| `MAX_TOKENS` | `800` | Max generation length |
| `TEMPERATURE` | `0.2` | Generation temperature |
| `LOG_LEVEL` | `info` | Logging verbosity |

## Verifying the Build

### Check Models are Present

```bash
docker run --rm rag-app:latest ls -lh /app/models/
docker run --rm rag-app:latest ls -lh /app/models/embeddings/
```

### Test Offline Validation

```bash
docker run --rm rag-app:latest python tools/validate_offline.py
```

### Check CUDA Support

```bash
docker run --rm --gpus all rag-app:latest nvidia-smi
docker run --rm --gpus all rag-app:latest python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Accessing the Application

Once running:

- **Web UI:** http://localhost:8080
- **API Docs:** http://localhost:8080/docs
- **Health Check:** http://localhost:8080/health

## Monitoring

### View Logs

```bash
# All logs
docker logs -f rag-app

# Last 100 lines
docker logs --tail 100 rag-app

# Since timestamp
docker logs --since 2024-01-01T00:00:00 rag-app
```

### Resource Usage

```bash
# CPU and memory usage
docker stats rag-app

# GPU usage
docker exec rag-app nvidia-smi

# Disk usage
docker system df
```

## Troubleshooting

### Build Fails During Model Download

**Issue:** Network timeout or disk space

**Solution:**
```bash
# Pre-download models locally
python tools/download_models.py --all

# Then build
docker build -t rag-app:latest .
```

### Container Exits Immediately

**Issue:** Missing data or configuration

**Solution:**
```bash
# Check logs
docker logs rag-app

# Ensure data is mounted
docker run --rm -v $(pwd)/data:/data rag-app:latest ls -l /data/
```

### GPU Not Detected

**Issue:** NVIDIA runtime not configured

**Solution:**
```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

**Issue:** Insufficient RAM/VRAM

**Solution:**
```bash
# Increase Docker memory limit
docker run -d --memory="16g" --gpus all rag-app:latest

# Or use CPU mode
docker run -d -e FORCE_CPU=true -e N_GPU_LAYERS=0 rag-app:latest
```

### Offline Validation Fails

**Issue:** Models not properly downloaded

**Solution:**
```bash
# Check what's missing
docker run --rm rag-app:latest python tools/validate_offline.py

# Rebuild with verbose output
docker build --progress=plain -t rag-app:latest . 2>&1 | tee build.log
```

## Production Deployment

### With Load Balancer

```bash
# Run multiple instances
docker run -d --name rag-app-1 -p 8081:8080 --gpus '"device=0"' rag-app:latest
docker run -d --name rag-app-2 -p 8082:8080 --gpus '"device=1"' rag-app:latest

# Use nginx for load balancing
```

### With Auto-Restart

```bash
docker run -d \
  --name rag-app \
  --restart unless-stopped \
  --gpus all \
  -p 8080:8080 \
  rag-app:latest
```

### With Logging Driver

```bash
docker run -d \
  --name rag-app \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  --gpus all \
  -p 8080:8080 \
  rag-app:latest
```

## Updating the Application

```bash
# Pull latest code
git pull

# Rebuild image
docker build -t rag-app:latest .

# Stop old container
docker stop rag-app
docker rm rag-app

# Start new container
docker run -d --name rag-app --gpus all -p 8080:8080 rag-app:latest
```

## Cleaning Up

```bash
# Stop and remove container
docker stop rag-app
docker rm rag-app

# Remove image
docker rmi rag-app:latest

# Clean up volumes
docker volume rm rag-index

# Clean up build cache
docker builder prune -a
```

## Security Considerations

### Run as Non-Root User

Add to Dockerfile:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### Scan for Vulnerabilities

```bash
docker scan rag-app:latest
```

### Limit Resources

```bash
docker run -d \
  --memory="8g" \
  --cpus="4" \
  --pids-limit=100 \
  rag-app:latest
```

## Performance Optimization

### Build Cache

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t rag-app:latest .

# Multi-stage build (if needed)
docker build --target production -t rag-app:latest .
```

### Runtime Performance

```bash
# Use shared memory for better performance
docker run -d \
  --shm-size=2g \
  --gpus all \
  -p 8080:8080 \
  rag-app:latest
```

---

**Last Updated:** 2025-10-11  
**Docker Version:** 20.10+  
**CUDA Version:** 12.4.0
