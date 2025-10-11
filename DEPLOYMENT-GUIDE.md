# 🚀 Deployment Guide - Activating Python AI Service

## Overview

This guide walks you through deploying the **real AI models** (CLIP, OCR, Face Detection) to move from simulated ML to production-grade inference.

---

## Step 1: Deploy Python AI Service

### Option A: Local Development (Quick Test)

```bash
# Navigate to Python service
cd python-service

# Run setup script (installs dependencies + downloads models)
./setup.sh

# This will:
# ✓ Create Python virtual environment
# ✓ Install PyTorch, FastAPI, CLIP, PaddleOCR, etc.
# ✓ Download CLIP ViT-L/14 model (~1.7GB)
# ✓ Verify GPU availability
# ✓ Test installations

# Time: ~10-15 minutes
```

**Start the service:**
```bash
# Activate virtual environment
source venv/bin/activate

# Start FastAPI server
uvicorn app.main:app --reload --port 8000

# Or with better logging
uvicorn app.main:app --reload --port 8000 --log-level info
```

**Verify it's running:**
```bash
# Health check
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "models": {
    "clip": true,
    "ocr": true,
    "face": true,
    "emotion": true
  },
  "device": "cuda",  // or "cpu"
  "gpu_available": true  // or false
}
```

---

### Option B: Docker (Production-Ready)

```bash
# Navigate to Python service
cd python-service

# Build and start with Docker Compose
docker-compose up --build

# Or run in background
docker-compose up --build -d

# This will:
# ✓ Build Docker image with all dependencies
# ✓ Download ML models inside container
# ✓ Start FastAPI on port 8000
# ✓ Enable GPU support (if available)
# ✓ Set up health checks
# ✓ Configure auto-restart

# Time: ~15-20 minutes (first build)
```

**Check status:**
```bash
# View logs
docker-compose logs -f inference-service

# Check health
curl http://localhost:8000/health

# Stop service
docker-compose down
```

---

## Step 2: Connect Next.js to Python Service

### Update Environment Variables

Create or update `.env.local` in the project root:

```bash
cd /Users/nicvenettacci/Desktop/Thumbnail\ Lab/thumbnail-lab

# Add these lines to .env.local
cat >> .env.local << EOF

# Python AI Service
USE_PYTHON_SERVICE=true
PYTHON_SERVICE_URL=http://localhost:8000
EOF
```

**Or manually edit `.env.local`:**
```bash
# Python AI Service Configuration
USE_PYTHON_SERVICE=true
PYTHON_SERVICE_URL=http://localhost:8000

# For Docker deployment
# PYTHON_SERVICE_URL=http://python-service:8000
```

### Restart Next.js

```bash
# Kill current server
pkill -f "next dev"

# Restart
npm run dev
```

**Verify connection:**
```bash
# Check Next.js logs - should see:
# "[Inference] Proxying to Python FastAPI service..."
```

---

## Step 3: Test Real AI Models

### Test Health

```bash
# Python service directly
curl http://localhost:8000/health

# Through Next.js proxy
curl -X POST http://localhost:3000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Video",
    "thumbnails": [
      {"id":"A","url":"https://picsum.photos/1280/720"}
    ]
  }'

# Check response header
# Should include: X-Inference-Backend: python
```

### Test via Web Interface

1. Visit `http://localhost:3000`
2. Click "Test Your Thumbnails"
3. Upload 3 images
4. Click "Analyze"
5. **Check browser console** - should see:
   ```
   [Inference] Proxying to Python FastAPI service...
   ```

6. **Check terminal** - Python service logs:
   ```
   [ModelPipeline] Initializing on device: cuda
   [ModelPipeline] ✓ CLIP ViT-L/14 loaded
   [ModelPipeline] ✓ PaddleOCR loaded
   [Inference] Processing 3 thumbnails...
   ```

---

## Step 4: Verify Real Models Are Active

### Check Model Status

```bash
# Query Python service
curl http://localhost:8000/health | python3 -m json.tool

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-10-11T12:00:00",
  "models": {
    "clip": true,      # ← Real CLIP loaded
    "ocr": true,       # ← Real PaddleOCR loaded
    "face": true,      # ← Real RetinaFace loaded
    "emotion": true,   # ← Real FER loaded
    "ranking": false   # ← Your trained model (add later)
  },
  "device": "cuda",    # or "cpu"
  "gpu_available": true
}
```

### Compare Results

**Before (TypeScript simulation):**
```json
{
  "metadata": {
    "analysisType": "ml_powered_two_stage",
    "version": "3.0.0"
  },
  "thumbnails": [
    {
      "ctr_score": 92,
      "subscores": {
        "clarity": 88  // Random/heuristic
      }
    }
  ]
}
```

**After (Python real AI):**
```json
{
  "metadata": {
    "processing_time_ms": 245,
    "model_version": "1.0.0",
    "device": "cuda"
  },
  "thumbnails": [
    {
      "ctr_score": 87.3,
      "subscores": {
        "clarity": 82  // From real OCR analysis
      }
    }
  ]
}
```

**Key differences:**
- ✅ Real OCR detects actual text
- ✅ Real face detection finds actual faces
- ✅ CLIP embeddings from real images
- ✅ Consistent results for same image
- ✅ More accurate predictions

---

## GPU Acceleration (Optional but Recommended)

### Check GPU Availability

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Check PyTorch can see GPU
python3 << EOF
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
EOF
```

### Enable GPU in Docker

```yaml
# python-service/docker-compose.yml (already configured)
services:
  inference-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Start with GPU:**
```bash
docker-compose up --build
```

**Performance improvement:**
- CPU: ~500ms per thumbnail
- GPU (T4): ~100ms per thumbnail
- GPU (A100): ~50ms per thumbnail

---

## Production Deployment

### Full Stack (Next.js + Python)

```bash
# Build both services
docker-compose -f docker-compose.production.yml up --build -d
```

**docker-compose.production.yml:**
```yaml
version: '3.8'

services:
  nextjs:
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - USE_PYTHON_SERVICE=true
      - PYTHON_SERVICE_URL=http://python-service:8000
    depends_on:
      - python-service
    restart: unless-stopped
  
  python-service:
    build:
      context: ./python-service
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
      - MODEL_PATH=/app/models
    volumes:
      - ./python-service/models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

**Deploy:**
```bash
docker-compose -f docker-compose.production.yml up -d
```

---

## Monitoring

### Check Services

```bash
# Check all running services
docker-compose ps

# View logs
docker-compose logs -f

# View Python service logs only
docker-compose logs -f python-service

# View Next.js logs only
docker-compose logs -f nextjs
```

### Performance Metrics

**Add Prometheus monitoring:**
```python
# In python-service/app/main.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

**Visit:** `http://localhost:8000/metrics`

---

## Troubleshooting

### Python service won't start

**Check logs:**
```bash
docker-compose logs python-service
```

**Common issues:**

**1. CUDA errors:**
```bash
# Check GPU drivers
nvidia-smi

# Update to CPU mode
# In docker-compose.yml, set: DEVICE=cpu
# Remove GPU reservation
```

**2. Models not downloading:**
```bash
# Pre-download models manually
cd python-service
source venv/bin/activate
python3 << EOF
import clip
model, preprocess = clip.load("ViT-L/14", device="cpu")
print("✓ CLIP downloaded")
EOF
```

**3. Out of memory:**
```bash
# Use smaller model
# In app/main.py, change to:
# clip.load("ViT-B/32")  # Smaller, faster

# Or increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8GB+
```

### Next.js can't connect to Python

**Check:**
```bash
# Test Python service directly
curl http://localhost:8000/health

# If working, check Next.js env
cat .env.local | grep PYTHON

# Should show:
# USE_PYTHON_SERVICE=true
# PYTHON_SERVICE_URL=http://localhost:8000
```

**Fix:**
```bash
# Restart Next.js after env changes
pkill -f "next dev"
npm run dev
```

---

## Current Status Check

### Quick Test Script

```bash
#!/bin/bash
echo "🔍 Checking Thumbnail Lab Status..."

# Check Next.js
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Next.js running on port 3000"
else
    echo "❌ Next.js not running"
fi

# Check Python service
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Python service running on port 8000"
    
    # Check which models are loaded
    curl -s http://localhost:8000/health | grep -o '"clip":[^,]*' | grep true && echo "  ✓ CLIP loaded" || echo "  ✗ CLIP not loaded"
    curl -s http://localhost:8000/health | grep -o '"ocr":[^,]*' | grep true && echo "  ✓ OCR loaded" || echo "  ✗ OCR not loaded"
    curl -s http://localhost:8000/health | grep -o '"face":[^,]*' | grep true && echo "  ✓ Face detection loaded" || echo "  ✗ Face not loaded"
else
    echo "❌ Python service not running"
fi

# Check mode
if grep -q "USE_PYTHON_SERVICE=true" .env.local 2>/dev/null; then
    echo "✅ Next.js configured to use Python service"
else
    echo "⚠️  Next.js using TypeScript simulation (Python not enabled)"
fi
```

Save as `check-status.sh` and run:
```bash
chmod +x check-status.sh
./check-status.sh
```

---

## Summary

### To Activate Real AI Models:

**1. Setup Python service:**
```bash
cd python-service
./setup.sh
```

**2. Start Python service:**
```bash
# Option A: Direct
source venv/bin/activate
uvicorn app.main:app --port 8000

# Option B: Docker (recommended)
docker-compose up --build
```

**3. Configure Next.js:**
```bash
# Add to .env.local
USE_PYTHON_SERVICE=true
PYTHON_SERVICE_URL=http://localhost:8000
```

**4. Restart Next.js:**
```bash
pkill -f "next dev"
npm run dev
```

**5. Test:**
```bash
curl http://localhost:8000/health
# Should show all models loaded: true
```

---

## What You Get

### Before (TypeScript):
- ⚡ Fast (50-150ms)
- 🎲 Simulated scores
- 📦 No dependencies

### After (Python + Real AI):
- 🎯 Accurate (real ML models)
- 🔬 Real OCR, face detection, CLIP
- 📊 Production-grade predictions
- 🚀 GPU accelerated (optional)
- ⏱️ 200-500ms (CPU) or 50-150ms (GPU)

---

**Ready to deploy? Follow the steps above!** 🚀

**Current status: TypeScript simulation (working perfectly)**  
**Next step: Activate Python service for real AI models**
