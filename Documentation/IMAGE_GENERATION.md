# Image Generation Configuration

## Overview

The RAG application supports three image generation providers:
1. **local** - Uses SDXL-Turbo model locally (requires GPU)
2. **pollinations** - Uses Pollinations.ai API (free, no API key needed)
3. **openai** - Uses OpenAI-compatible API (requires API key)

## Provider Selection

The application supports three independent image generation providers that work regardless of GPU availability:

### Provider Behaviors

1. **local** - Requires GPU for optimal performance
   - Uses local SDXL-Turbo model
   - Can work on CPU but will be very slow
   - Requires ~7GB model storage

2. **pollinations** - Works everywhere (CPU/GPU)
   - Free external API service
   - No API key needed
   - Requires internet connection
   - **Recommended for Docker deployments**

3. **openai** - Works everywhere (CPU/GPU)
   - Uses OpenAI-compatible API
   - Requires API key
   - Requires internet connection

### Example Configurations

#### Scenario 1: Docker without GPU (Recommended)
```bash
# Configuration
FORCE_CPU=true
IMAGE_API_PROVIDER=pollinations

# Result
✅ Image generation works via Pollinations.ai
✅ No GPU required
✅ No API key needed
```

#### Scenario 2: Docker with GPU (Best Performance)
```bash
# Configuration
FORCE_CPU=false
N_GPU_LAYERS=35
IMAGE_API_PROVIDER=local

# Result
✅ Image generation works via local SDXL-Turbo
✅ Fastest generation
✅ No external dependencies
```

#### Scenario 3: CPU-only with OpenAI API
```bash
# Configuration
FORCE_CPU=true
IMAGE_API_PROVIDER=openai
IMAGE_API_KEY=sk-proj-xxxxx

# Result
✅ Image generation works via OpenAI API
✅ High-quality images
✅ No GPU required
```

## Recommended Configurations

### Docker Deployment (CPU-only)
```env
FORCE_CPU=true
IMAGE_API_PROVIDER=pollinations
```
✅ No additional model storage needed
✅ No API key required
✅ Works immediately

### Docker with GPU
```env
FORCE_CPU=false
N_GPU_LAYERS=35
IMAGE_API_PROVIDER=local
IMAGE_MODEL_PATH=/app/models/sdxl-turbo
```
✅ Fastest image generation
✅ No external API dependency
⚠️ Requires ~7GB additional Docker image size

### Production with OpenAI API
```env
FORCE_CPU=true
IMAGE_API_PROVIDER=openai
IMAGE_API_KEY=your-api-key-here
```
✅ High-quality images
✅ No local model storage
⚠️ Requires API key and costs per image

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_API_PROVIDER` | `pollinations` | Provider: `local`, `pollinations`, or `openai` |
| `IMAGE_API_KEY` | `""` | API key for OpenAI-compatible endpoints |
| `IMAGE_MODEL_PATH` | `/app/models/sdxl-turbo` | Local model cache directory |
| `IMAGE_MODEL_NAME` | `stabilityai/sdxl-turbo` | Hugging Face model identifier |
| `IMAGE_INFERENCE_STEPS` | `100` | Number of diffusion steps |
| `IMAGE_GUIDANCE_SCALE` | `12.0` | Guidance scale for generation |
| `FORCE_CPU` | `true` | Force CPU usage (affects auto-fallback) |

## Docker Examples

### CPU-only deployment (recommended)
```bash
docker run -d \
  --name rag-app \
  -p 8080:8080 \
  -e FORCE_CPU=true \
  -e IMAGE_API_PROVIDER=pollinations \
  genai-rag-app:latest
```

### GPU-enabled deployment
```bash
docker run -d \
  --name rag-app \
  --gpus all \
  -p 8080:8080 \
  -e FORCE_CPU=false \
  -e N_GPU_LAYERS=35 \
  -e IMAGE_API_PROVIDER=local \
  genai-rag-app:latest
```

### With OpenAI API
```bash
docker run -d \
  --name rag-app \
  -p 8080:8080 \
  -e IMAGE_API_PROVIDER=openai \
  -e IMAGE_API_KEY=sk-proj-xxxxx \
  genai-rag-app:latest
```

## Troubleshooting

### Local model fails to load
**Problem**: Error loading local SDXL model

**Solution**: Switch to an external provider:
1. Set `IMAGE_API_PROVIDER=pollinations` (free, no key needed)
2. Or set `IMAGE_API_PROVIDER=openai` with valid `IMAGE_API_KEY`
3. Rebuild/restart the application

### Pollinations API timeout
**Problem**: External API is slow or unavailable

**Solution**: Set a longer `REQUEST_TIMEOUT` or switch to local model with GPU

### OpenAI API authentication error
**Problem**: Invalid API key

**Solution**: 
1. Verify `IMAGE_API_KEY` is set correctly
2. Check API key has image generation permissions
3. Fallback to `pollinations` provider (no key needed)

## Performance Comparison

| Provider | Speed | Quality | Cost | Requirements |
|----------|-------|---------|------|--------------|
| **local** | ⚡ Fastest | ⭐⭐⭐⭐ | 💰 Free | GPU + 7GB storage |
| **pollinations** | 🐢 Slow | ⭐⭐⭐ | 💰 Free | Internet only |
| **openai** | 🚀 Fast | ⭐⭐⭐⭐⭐ | 💰💰 Paid | API key + Internet |

## Notes

- The auto-fallback feature ensures the application always has a working image generation method
- Pollinations.ai is recommended for Docker deployments without GPU
- Local generation provides best performance but requires GPU and additional storage
- Auto-configuration happens at application startup and is logged
