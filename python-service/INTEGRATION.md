# Integration Guide: Python FastAPI + Next.js

## Overview

This guide shows how to integrate the Python FastAPI inference service with your Next.js Thumbnail Lab application.

## Architecture Options

### Option 1: Hybrid (Recommended for Production)

```
User → Next.js (Port 3000) → Python FastAPI (Port 8000) → ML Models
       ↑                      ↑
       │                      │
       UI/UX                  Inference
```

**Benefits:**
- Best of both worlds
- Fast UI with React
- High-performance ML with Python
- Easy to scale independently

### Option 2: Standalone Python API

```
User → Python FastAPI (Port 8000) → ML Models
       ↑
       Direct API access
```

**Benefits:**
- Simpler deployment
- Lower latency
- Good for API-only use cases

### Option 3: TypeScript Only

```
User → Next.js (Port 3000) → Simulated ML
```

**Benefits:**
- No Python dependencies
- Faster prototyping
- Good for demos

## Setup Steps

### 1. Install Python Service

```bash
cd python-service
chmod +x setup.sh
./setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies
- Download CLIP and other models
- Verify installation

### 2. Start Python Service

```bash
# Activate virtual environment
source venv/bin/activate

# Start server
uvicorn app.main:app --reload --port 8000
```

Or with Docker:
```bash
docker-compose up --build
```

### 3. Configure Next.js to Use Python Service

Add to `.env.local`:
```bash
USE_PYTHON_SERVICE=true
PYTHON_SERVICE_URL=http://localhost:8000
```

### 4. Test Integration

```bash
# Test Python service directly
curl http://localhost:8000/health

# Test through Next.js proxy
curl -X POST http://localhost:3000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Video",
    "thumbnails": [
      {"id":"A","url":"https://picsum.photos/1280/720"}
    ]
  }'
```

## Development Workflow

### Local Development

**Terminal 1 - Next.js:**
```bash
cd thumbnail-lab
npm run dev
# Runs on http://localhost:3000
```

**Terminal 2 - Python Service:**
```bash
cd python-service
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
# Runs on http://localhost:8000
```

**Terminal 3 - Testing:**
```bash
# Test the full stack
curl -X POST http://localhost:3000/api/v1/score \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### Hot Reload

Both services support hot reload:
- **Next.js**: Automatic on file save
- **Python**: Use `--reload` flag with uvicorn

## API Integration Patterns

### Pattern 1: Proxy (Current Implementation)

Next.js API routes proxy to Python service when enabled:

```typescript
// app/api/v1/score/route.ts
if (USE_PYTHON_SERVICE) {
  const response = await fetch(`${PYTHON_SERVICE_URL}/v1/score`, {
    method: 'POST',
    body: JSON.stringify(body)
  });
  return NextResponse.json(await response.json());
}
```

**Pros:**
- Single endpoint for frontend
- Automatic fallback to TypeScript
- Easy to switch backends

**Cons:**
- Extra network hop
- Slightly higher latency

### Pattern 2: Direct Client-Side

Frontend calls Python service directly:

```typescript
// In your React component
const response = await fetch('http://localhost:8000/v1/score', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ title, thumbnails })
});
```

**Pros:**
- Lower latency
- Direct communication

**Cons:**
- CORS configuration required
- Multiple endpoints to manage

### Pattern 3: Next.js Server Actions

Use Next.js server actions to call Python service:

```typescript
'use server'

export async function scoreThumbnails(title: string, thumbnails: any[]) {
  const response = await fetch('http://localhost:8000/v1/score', {
    method: 'POST',
    body: JSON.stringify({ title, thumbnails })
  });
  return await response.json();
}
```

## Production Deployment

### Scenario 1: Same Server

Deploy both services on the same machine:

```yaml
# docker-compose.production.yml
version: '3.8'
services:
  nextjs:
    build: ../
    ports:
      - "3000:3000"
    environment:
      - USE_PYTHON_SERVICE=true
      - PYTHON_SERVICE_URL=http://python-service:8000
    depends_on:
      - python-service
  
  python-service:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Scenario 2: Separate Servers

Deploy services independently:

**Python Service (GPU Server):**
```bash
# On ML server with GPU
cd python-service
docker-compose up -d
# Exposes port 8000
```

**Next.js (Web Server):**
```bash
# On web server
cd thumbnail-lab
npm run build
npm start
# Set PYTHON_SERVICE_URL=https://ml.example.com
```

### Scenario 3: Serverless

**Next.js on Vercel:**
```bash
vercel deploy
```

**Python on AWS Lambda / Modal:**
```python
# Use Modal for serverless GPU inference
import modal

stub = modal.Stub("thumbnail-scorer")

@stub.function(gpu="T4", image=modal.Image.debian_slim().pip_install_from_requirements("requirements.txt"))
def score_thumbnail(title: str, thumbnail_url: str):
    # Your inference code
    pass
```

## Performance Optimization

### 1. Caching

Add Redis for embedding caching:

```python
import redis
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_embedding(image_url: str):
    cache_key = f"emb:{hashlib.md5(image_url.encode()).hexdigest()}"
    cached = redis_client.get(cache_key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)
    return None

def cache_embedding(image_url: str, embedding: np.ndarray):
    cache_key = f"emb:{hashlib.md5(image_url.encode()).hexdigest()}"
    redis_client.setex(cache_key, 3600, embedding.tobytes())
```

### 2. Batch Processing

Process multiple thumbnails in parallel:

```python
import asyncio

async def process_thumbnails_parallel(thumbnails: List[Thumb], title: str):
    tasks = [extract_features_async(t.url, title) for t in thumbnails]
    return await asyncio.gather(*tasks)
```

### 3. Model Optimization

**Quantization:**
```python
# Convert to FP16 for 2x speedup on GPU
model.half()
```

**ONNX Export:**
```python
torch.onnx.export(model, dummy_input, "model.onnx")
# Then use ONNX Runtime for faster inference
```

**TensorRT (NVIDIA GPUs):**
```python
import tensorrt as trt
# Convert PyTorch → TensorRT for 3-5x speedup
```

## Monitoring

### Logging

Add structured logging:

```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Use in code
logger.info("Processing thumbnail", extra={
    "thumbnail_id": "A",
    "ctr_score": 85.3,
    "duration_ms": 145
})
```

### Metrics

Add Prometheus metrics:

```python
from prometheus_fastapi_instrumentator import Instrumentator

# Add to main.py
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram

inference_counter = Counter('thumbnail_inferences_total', 'Total inferences')
inference_duration = Histogram('thumbnail_inference_duration_seconds', 'Inference duration')

@inference_duration.time()
def score(req: ScoreRequest):
    inference_counter.inc()
    # ... your code
```

## Troubleshooting

### Python service not responding

```bash
# Check if service is running
curl http://localhost:8000/health

# Check logs
docker-compose logs -f python-service

# Restart service
docker-compose restart python-service
```

### Models not loading

```bash
# Test model loading manually
python3 << EOF
import clip
model, preprocess = clip.load("ViT-L/14")
print("CLIP loaded successfully")
EOF
```

### CORS errors

Update CORS settings in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Add your production domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Testing

### Unit Tests

```python
# tests/test_inference.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_score():
    response = client.post("/v1/score", json={
        "title": "Test Video",
        "thumbnails": [
            {"id": "A", "url": "https://picsum.photos/1280/720"}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "winner_id" in data
    assert data["winner_id"] == "A"
```

Run tests:
```bash
pytest tests/ -v
```

## Next Steps

1. ✅ Setup Python service
2. ✅ Verify models load correctly
3. ✅ Test `/v1/score` endpoint
4. ✅ Integrate with Next.js
5. ⏳ Train and load your custom ranking model
6. ⏳ Set up production deployment
7. ⏳ Add monitoring and logging
8. ⏳ Configure GPU acceleration
9. ⏳ Implement caching layer
10. ⏳ Scale with load balancing

---

**Your ML inference service is ready to plug in!** 🎯

