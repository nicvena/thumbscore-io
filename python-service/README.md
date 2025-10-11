# Thumbnail Lab Python Service

High-performance Python backend for automated YouTube thumbnail collection and ML-powered scoring.

## Features

- 🤖 **Automated Collection**: Fetches trending YouTube thumbnails daily
- 🧠 **CLIP Embeddings**: Computes semantic embeddings for similarity search
- 📊 **ML Scoring**: Advanced thumbnail analysis and CTR prediction
- ⏰ **Scheduled Jobs**: APScheduler for background tasks
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

## Automated Collection

The service automatically collects trending thumbnails every day at 3 AM UTC:

- **5 Niches**: Tech, Gaming, Education, Entertainment, People & Blogs
- **30 Videos per niche** (150 total per day)
- **CLIP embeddings** computed for each thumbnail
- **Views per hour** calculated from publication time
- **Auto-cleanup** removes entries older than 90 days

## Testing

### Test Collection System
```bash
python test_collection.py
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Manual refresh
curl http://localhost:8000/internal/refresh-library

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
│   ├── main.py                 # FastAPI app with scheduler
│   ├── features.py             # CLIP encoding utilities
│   ├── ref_library.py          # Similarity search
│   └── tasks/
│       └── collect_thumbnails.py  # Automated collection
├── requirements.txt            # Dependencies
├── test_collection.py         # Test script
└── README.md                  # This file
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