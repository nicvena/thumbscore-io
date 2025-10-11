# FastAPI Inference Service

High-performance Python ML service that integrates with your Next.js Thumbnail Lab app.

## Overview

This FastAPI service provides:
- **Real ML model inference** (CLIP, OCR, Face Detection, Emotion)
- **Feature extraction pipeline** (all visual features)
- **Production-ready API** matching the `/v1/score` contract
- **GPU acceleration** support
- **Easy Docker deployment**

## Architecture

```
┌─────────────────┐
│  Next.js App    │  ← User-facing web application
│  (Port 3000)    │
└────────┬────────┘
         │
         │ HTTP
         ↓
┌─────────────────┐
│  FastAPI        │  ← ML inference service
│  (Port 8000)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  ML Models      │
│  - CLIP         │
│  - PaddleOCR    │
│  - RetinaFace   │
│  - FER          │
└─────────────────┘
```

## Quick Start

### Option 1: Local Development

```bash
# Create virtual environment
cd python-service
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python -m app.main
# OR
uvicorn app.main:app --reload --port 8000
```

### Option 2: Docker

```bash
# Build and run with Docker Compose
cd python-service
docker-compose up --build

# Or with plain Docker
docker build -t thumbnail-inference .
docker run -p 8000:8000 thumbnail-inference
```

### Option 3: Production Deployment

```bash
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 60
```

## Testing

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models": {
    "clip": true,
    "ocr": true,
    "face": true,
    "emotion": true,
    "ranking": false
  },
  "device": "cpu",
  "gpu_available": false
}
```

### 2. Score Thumbnails

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "title": "I Tried MrBeasts $1 vs $100,000 Plane Seat",
    "thumbnails": [
      {"id":"A","url":"https://picsum.photos/1280/720"},
      {"id":"B","url":"https://picsum.photos/1280/721"},
      {"id":"C","url":"https://picsum.photos/1280/722"}
    ],
    "category":"people-blogs"
  }'
```

### 3. Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/score",
    json={
        "title": "Amazing Video Title",
        "thumbnails": [
            {"id": "A", "url": "https://example.com/thumb_a.jpg"},
            {"id": "B", "url": "https://example.com/thumb_b.jpg"}
        ],
        "category": "education"
    }
)

data = response.json()
print(f"Winner: {data['winner_id']}")
print(f"Score: {data['thumbnails'][0]['ctr_score']}")
print(f"Insights: {data['thumbnails'][0]['insights']}")
```

## Integration with Next.js App

### Update Next.js API to proxy to Python service:

```typescript
// app/api/v1/score/route.ts (add proxy option)

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  const USE_PYTHON_SERVICE = process.env.USE_PYTHON_SERVICE === 'true';
  
  if (USE_PYTHON_SERVICE) {
    // Proxy to Python FastAPI service
    const body = await request.json();
    const response = await fetch(`${PYTHON_SERVICE_URL}/v1/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    
    const data = await response.json();
    return NextResponse.json(data);
  }
  
  // Otherwise use TypeScript simulation
  // ... existing code
}
```

### Environment Variables

```bash
# .env.local (Next.js)
USE_PYTHON_SERVICE=true
PYTHON_SERVICE_URL=http://localhost:8000
```

## Model Loading

### 1. CLIP

CLIP will download automatically on first run (~1.7GB for ViT-L/14):

```python
import clip
model, preprocess = clip.load("ViT-L/14", device="cuda")
```

### 2. Your Trained Model

Place your trained ranking model in `models/` directory:

```python
# In main.py startup
pipeline.ranking_model = torch.load("models/ranking_model.pt", map_location=device)
pipeline.ranking_model.eval()
```

### 3. Model Directory Structure

```
python-service/
├── app/
│   └── main.py
├── models/
│   ├── ranking_model.pt          # Your trained model
│   ├── clip_finetuned.pt         # Fine-tuned CLIP (optional)
│   └── metadata.json             # Model metadata
└── requirements.txt
```

## Performance

### Typical Latency (CPU)
- **1 thumbnail**: ~200-400ms
- **3 thumbnails**: ~500-1000ms
- **10 thumbnails**: ~2-4 seconds

### With GPU (NVIDIA CUDA)
- **1 thumbnail**: ~50-100ms
- **3 thumbnails**: ~100-200ms
- **10 thumbnails**: ~300-600ms

### Optimization Tips

1. **Batch processing** - Process multiple thumbnails in parallel
2. **Caching** - Cache CLIP embeddings by image URL
3. **Model quantization** - Reduce model size for faster inference
4. **Multiple workers** - Scale horizontally with gunicorn

## GPU Setup

### Enable CUDA Support

```bash
# Install CUDA-enabled PyTorch
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```

### Docker with GPU

```yaml
# docker-compose.yml
services:
  inference-service:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - DEVICE=cuda
```

## API Documentation

Once running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **OpenAPI spec**: http://localhost:8000/openapi.json

## Monitoring

### Logs

```bash
# View logs
docker-compose logs -f inference-service

# Or with plain Docker
docker logs -f thumbnail-inference
```

### Metrics

Add Prometheus metrics:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Visit http://localhost:8000/metrics for Prometheus-compatible metrics.

## Troubleshooting

### Models not loading

**Issue**: Models fail to download or load

**Solution**:
```bash
# Pre-download models
python -c "import clip; clip.load('ViT-L/14')"
python -c "from paddleocr import PaddleOCR; PaddleOCR()"
```

### Out of memory

**Issue**: GPU/CPU runs out of memory

**Solution**:
- Reduce batch size
- Use smaller CLIP model (ViT-B/32)
- Enable model quantization

### Slow inference

**Issue**: High latency per request

**Solution**:
- Enable GPU acceleration
- Add Redis caching for embeddings
- Use model optimization (ONNX, TensorRT)

## Development

### Project Structure

```
python-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # Main FastAPI application
│   ├── models.py            # Pydantic models (optional)
│   ├── feature_extraction.py  # Feature extraction logic (optional)
│   └── inference.py         # Model inference logic (optional)
├── models/                  # Trained model weights
├── tests/                   # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Add Your Custom Model

Replace the `model_predict` function in `main.py`:

```python
def model_predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Replace with your trained model"""
    
    with torch.no_grad():
        # 1. Prepare input tensor
        embedding = torch.from_numpy(features['clip_embedding']).to(pipeline.device)
        
        # 2. Run your model
        ctr_score, subscores = pipeline.ranking_model(
            embedding,
            features['ocr'],
            features['faces'],
            features['colors']
        )
        
        # 3. Return predictions
        return {
            "ctr_score": float(ctr_score),
            "subscores": {
                "clarity": int(subscores[0]),
                "subject_prominence": int(subscores[1]),
                "contrast_pop": int(subscores[2]),
                "emotion": int(subscores[3]),
                "hierarchy": int(subscores[4]),
                "title_match": int(subscores[5])
            }
        }
```

## Deployment Checklist

- [ ] GPU setup (if using CUDA)
- [ ] Load trained model weights
- [ ] Configure environment variables
- [ ] Set up monitoring (logs, metrics)
- [ ] Configure CORS for your domain
- [ ] Add rate limiting
- [ ] Enable caching (Redis)
- [ ] Set up health checks
- [ ] Configure autoscaling
- [ ] Add authentication (API keys)

## Environment Variables

```bash
# .env
DEVICE=cuda                    # cpu or cuda
MODEL_PATH=/app/models
CLIP_MODEL=ViT-L/14           # ViT-L/14 or ViT-B/32
ENABLE_CACHING=true
REDIS_URL=redis://localhost:6379
LOG_LEVEL=info
MAX_WORKERS=4
TIMEOUT_SECONDS=60
```

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Test health endpoint: `curl http://localhost:8000/health`
- Review FastAPI docs: http://localhost:8000/docs
- Open an issue on GitHub

---

**Ready to serve production-grade ML inference!** 🚀

