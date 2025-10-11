# Thumbnail Lab Python Service

High-performance Python backend for automated YouTube thumbnail collection and ML-powered scoring.

## Features

- 🤖 **Automated Collection**: Fetches trending YouTube thumbnails daily
- 🧠 **CLIP Embeddings**: Computes semantic embeddings for similarity search
- 🚀 **FAISS Indices**: Fast similarity search with nightly index rebuilding
- 📊 **ML Scoring**: Advanced thumbnail analysis and CTR prediction
- ⏰ **Scheduled Jobs**: APScheduler for background tasks (Hobart timezone)
- 🔄 **Auto-cleanup**: Removes old thumbnails (90+ days)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Required for YouTube API
export YOUTUBE_API_KEY="your_youtube_api_key"

# Required for Supabase database
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_key"

# Optional: GPU acceleration
export CUDA_VISIBLE_DEVICES="0"
```

### 3. Create Supabase Table

Run this SQL in your Supabase SQL editor:

```sql
CREATE TABLE ref_thumbnails (
    video_id TEXT PRIMARY KEY,
    niche TEXT NOT NULL,
    title TEXT NOT NULL,
    thumbnail_url TEXT NOT NULL,
    views_per_hour FLOAT NOT NULL,
    view_count INTEGER NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    channel_title TEXT,
    description TEXT,
    embedding VECTOR(768) NOT NULL,
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for similarity search
CREATE INDEX ON ref_thumbnails USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for niche queries
CREATE INDEX idx_ref_thumbnails_niche ON ref_thumbnails(niche);

-- Create index for cleanup
CREATE INDEX idx_ref_thumbnails_collected_at ON ref_thumbnails(collected_at);
```

### 4. Start the Service

```bash
# Development
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Manual Library Refresh
```bash
GET http://localhost:8000/internal/refresh-library
```

### Manual Index Rebuilding
```bash
GET http://localhost:8000/internal/rebuild-indices
```

### Index Statistics
```bash
GET http://localhost:8000/internal/index-stats
```

### Thumbnail Scoring
```bash
POST http://localhost:8000/v1/score
Content-Type: application/json

{
  "title": "My Video Title",
  "thumbnails": [
    {"id": "thumb1", "url": "https://example.com/thumb1.jpg"},
    {"id": "thumb2", "url": "https://example.com/thumb2.jpg"}
  ],
  "category": "tech"
}
```

## Automated Collection & Indexing

The service automatically collects and indexes trending thumbnails on Hobart time:

### Daily Schedule (Australia/Hobart timezone):
- **3:00 AM**: Collect trending YouTube thumbnails + Rebuild FAISS indices (combined operation)

### Collection Details:
- **5 Niches**: Tech, Gaming, Education, Entertainment, People & Blogs
- **30 Videos per niche** (150 total per day)
- **CLIP embeddings** computed for each thumbnail
- **Views per hour** calculated from publication time
- **Auto-cleanup** removes entries older than 90 days

### FAISS Indexing:
- **Fast similarity search** using FAISS (Facebook AI Similarity Search)
- **Separate indices** for each niche category (stored in `faiss_indices/` directory)
- **Cosine similarity** for semantic matching (IndexFlatIP)
- **Persistent storage** with automatic loading and caching
- **Thread-safe** global cache for production use
- **Instant lookups** in `/v1/score` endpoint via `ref_library.py`
- **Automatic rebuilding** after library refresh

## Testing

### Test Collection System
```bash
python test_collection.py
```

### Test FAISS Index System
```bash
python test_indices.py
```

### Test Complete FAISS Integration
```bash
python test_faiss_integration.py
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Manual refresh
curl http://localhost:8000/internal/refresh-library

# Manual index rebuilding
curl http://localhost:8000/internal/rebuild-indices

# Index statistics
curl http://localhost:8000/internal/index-stats

# Score thumbnails
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Video",
    "thumbnails": [
      {"id": "test", "url": "https://example.com/test.jpg"}
    ]
  }'
```

## Architecture

```
python-service/
├── app/
│   ├── main.py                     # FastAPI app with scheduler
│   ├── features.py                 # CLIP encoding utilities
│   ├── indices.py                  # Legacy FAISS index management
│   ├── ref_library.py              # FAISS similarity search & caching
│   └── tasks/
│       ├── collect_thumbnails.py   # Automated collection
│       └── build_faiss_index.py    # FAISS index builder
├── faiss_indices/                  # FAISS index storage (created at runtime)
│   ├── tech.index                  # Tech niche index
│   ├── tech_ids.npy                # Tech video ID mapping
│   ├── gaming.index                # Gaming niche index
│   └── ...                         # Other niche indices
├── requirements.txt                # Dependencies
├── test_collection.py             # Collection test script
├── test_indices.py                # Index test script
├── test_faiss_integration.py      # Complete integration test
└── README.md                      # This file
```

## FAISS Integration Flow

### How It Works:

1. **Library Refresh** (`collect_thumbnails.py`):
   - Fetches trending videos from YouTube API
   - Downloads thumbnails and computes CLIP embeddings
   - Stores in Supabase `ref_thumbnails` table

2. **Index Building** (`build_faiss_index.py`):
   - Reads embeddings from Supabase for each niche
   - Creates FAISS IndexFlatIP (cosine similarity)
   - Normalizes embeddings for proper cosine distance
   - Saves indices to `faiss_indices/{niche}.index`
   - Saves video ID mappings to `faiss_indices/{niche}_ids.npy`

3. **Similarity Search** (`ref_library.py`):
   - Loads indices into memory (with thread-safe caching)
   - Performs instant similarity lookups for `/v1/score` endpoint
   - Falls back to Supabase if index not available
   - Returns percentile scores (0-100)

### Usage in `/v1/score` endpoint:

```python
from app.ref_library import get_similarity_score

# User uploads thumbnail, we compute embedding
user_embedding = clip_encode(user_image)

# Get similarity score against reference library
niche = "tech"
similarity_percentile = get_similarity_score(user_embedding, niche)

# Use in scoring logic
print(f"Your thumbnail is in the {similarity_percentile:.1f}th percentile for {niche}")
```

### Environment Variables:

```bash
# Required
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
YOUTUBE_API_KEY=your_youtube_key

# Optional
FAISS_INDEX_PATH=faiss_indices  # Default: faiss_indices/
```

## Production Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production
```bash
# Core
YOUTUBE_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

# Optional
LOG_LEVEL=INFO
WORKERS=4
HOST=0.0.0.0
PORT=8000
```

## Monitoring

The service provides comprehensive logging:

- **Collection stats**: Videos fetched, thumbnails stored, cleanup count
- **Error handling**: Graceful failures with detailed error messages
- **Performance metrics**: Processing time and throughput
- **Health checks**: Model loading status and scheduler state

## Troubleshooting

### Common Issues

1. **YouTube API Quota Exceeded**
   - Check your API quota in Google Cloud Console
   - Reduce `VIDEOS_PER_NICHE` in `collect_thumbnails.py`

2. **CLIP Model Loading Failed**
   - Ensure sufficient disk space (CLIP model is ~2GB)
   - Check internet connection for model download

3. **Supabase Connection Issues**
   - Verify `SUPABASE_URL` and `SUPABASE_KEY`
   - Check if table exists and has correct schema

4. **Memory Issues**
   - Reduce batch sizes in collection
   - Use GPU if available for faster processing

### Logs
```bash
# View logs in real-time
tail -f /var/log/thumbnail-lab.log

# Check scheduler status
curl http://localhost:8000/health | jq '.scheduler'
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.