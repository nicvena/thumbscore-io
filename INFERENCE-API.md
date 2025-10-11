# Thumbnail Scoring Inference API v1

## Overview

Production-ready REST API for scoring and ranking YouTube thumbnails. **Drop-in service** that integrates into your existing application with a simple HTTP interface.

## Base URL

```
http://localhost:3000/api/v1
```

## Authentication

Currently open for development. Production deployment should add API key authentication:

```http
Authorization: Bearer YOUR_API_KEY
```

---

## Endpoints

### 1. Score Thumbnails

**`POST /v1/score`**

Score and rank multiple thumbnail options for a video.

#### Request

```json
{
  "title": "I Tried MrBeast's $1 vs $100,000 Plane Seat",
  "thumbnails": [
    {"id": "A", "url": "https://example.com/thumb_a.jpg"},
    {"id": "B", "url": "https://example.com/thumb_b.jpg"},
    {"id": "C", "url": "https://example.com/thumb_c.jpg"}
  ],
  "category": "people-blogs"
}
```

**Parameters:**
- `title` (string, required) - Video title for semantic matching
- `thumbnails` (array, required) - List of thumbnail options
  - `id` (string) - Unique identifier for thumbnail
  - `url` (string) - Public URL to thumbnail image
- `category` (string, optional) - Video category (e.g., "people-blogs", "gaming", "tech")

#### Response

```json
{
  "winner_id": "B",
  "thumbnails": [
    {
      "id": "B",
      "ctr_score": 85.3,
      "subscores": {
        "clarity": 82,
        "subject_prominence": 89,
        "contrast_pop": 86,
        "emotion": 88,
        "hierarchy": 83,
        "title_match": 84
      },
      "insights": [
        "Excellent face prominence drives attention",
        "High contrast text ensures mobile readability",
        "Strong title-thumbnail semantic alignment"
      ],
      "overlays": {
        "saliency_heatmap_url": "/api/v1/overlays/session_id/B/heatmap.png",
        "ocr_boxes_url": "/api/v1/overlays/session_id/B/ocr.png",
        "face_boxes_url": "/api/v1/overlays/session_id/B/faces.png"
      }
    },
    {
      "id": "A",
      "ctr_score": 71.2,
      "subscores": {
        "clarity": 68,
        "subject_prominence": 74,
        "contrast_pop": 63,
        "emotion": 59,
        "hierarchy": 66,
        "title_match": 72
      },
      "insights": [
        "Increase subject size ~25%",
        "Reduce words from 7→3; use bold block font",
        "Boost contrast between text and background"
      ],
      "overlays": {
        "saliency_heatmap_url": "/api/v1/overlays/session_id/A/heatmap.png",
        "ocr_boxes_url": "/api/v1/overlays/session_id/A/ocr.png",
        "face_boxes_url": "/api/v1/overlays/session_id/A/faces.png"
      }
    },
    {
      "id": "C",
      "ctr_score": 69.8,
      "subscores": {
        "clarity": 65,
        "subject_prominence": 72,
        "contrast_pop": 61,
        "emotion": 57,
        "hierarchy": 68,
        "title_match": 70
      },
      "insights": [
        "Increase text size and use high-contrast block font",
        "Boost saturation by 15-20% for more visual pop",
        "Add more expressive facial emotion or action"
      ],
      "overlays": {
        "saliency_heatmap_url": "/api/v1/overlays/session_id/C/heatmap.png",
        "ocr_boxes_url": "/api/v1/overlays/session_id/C/ocr.png",
        "face_boxes_url": "/api/v1/overlays/session_id/C/faces.png"
      }
    }
  ],
  "explanation": "B wins due to larger face prominence, high text contrast, and clear title alignment."
}
```

**Response Fields:**
- `winner_id` - ID of the highest-scoring thumbnail
- `thumbnails` - Array of scored thumbnails (sorted by `ctr_score` descending)
  - `ctr_score` - Predicted CTR/engagement score (0-100)
  - `subscores` - Breakdown of 6 interpretable metrics (0-100 each)
    - `clarity` - Text readability and contrast
    - `subject_prominence` - Face/subject size and positioning
    - `contrast_pop` - Color vibrancy and contrast
    - `emotion` - Emotional expression intensity
    - `hierarchy` - Visual composition and balance
    - `title_match` - Semantic alignment with title
  - `insights` - Actionable recommendations (prioritized)
  - `overlays` - URLs to visualization overlays
- `explanation` - Human-readable reason for winner selection

#### Response Headers

```
X-Processing-Time-Ms: 145
X-Model-Version: v3.0.0
X-Model-Type: two-stage-multi-task
```

---

### 2. Get API Info

**`GET /v1/score`**

Get API documentation and service status.

#### Response

```json
{
  "service": "Thumbnail Scoring Inference API",
  "version": "v1",
  "status": "operational",
  "model": {
    "type": "Two-Stage Multi-Task Learning",
    "version": "3.0.0",
    "features": [...]
  },
  "endpoints": {...},
  "limits": {...}
}
```

---

### 3. Get Visualization Overlays

**`GET /v1/overlays/:session/:id/:type`**

Retrieve visualization overlays for analyzed thumbnails.

#### Parameters
- `session` - Session ID from score response
- `id` - Thumbnail ID
- `type` - One of: `heatmap.png`, `ocr.png`, `faces.png`

#### Response Types

**Heatmap (`heatmap.png`):**
- Saliency map showing attention hotspots
- Rule of thirds alignment
- Focal point intensity overlay

**OCR (`ocr.png`):**
- Text detection bounding boxes
- Confidence scores
- Contrast analysis visualization

**Faces (`faces.png`):**
- Face detection boxes
- Emotion classification overlays
- Expression intensity heatmaps

---

## Code Examples

### cURL

```bash
curl -X POST http://localhost:3000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "title": "How I Built a $1M Business in 30 Days",
    "thumbnails": [
      {"id":"A","url":"https://cdn.example.com/thumb_a.jpg"},
      {"id":"B","url":"https://cdn.example.com/thumb_b.jpg"}
    ],
    "category":"education"
  }'
```

### JavaScript/TypeScript

```typescript
const response = await fetch('http://localhost:3000/api/v1/score', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: 'How I Built a $1M Business in 30 Days',
    thumbnails: [
      { id: 'A', url: 'https://cdn.example.com/thumb_a.jpg' },
      { id: 'B', url: 'https://cdn.example.com/thumb_b.jpg' }
    ],
    category: 'education'
  })
});

const data = await response.json();
console.log(`Winner: ${data.winner_id} with score ${data.thumbnails[0].ctr_score}`);

// Use insights
data.thumbnails[0].insights.forEach(insight => {
  console.log(`- ${insight}`);
});
```

### Python

```python
import requests

response = requests.post(
    'http://localhost:3000/api/v1/score',
    json={
        'title': 'How I Built a $1M Business in 30 Days',
        'thumbnails': [
            {'id': 'A', 'url': 'https://cdn.example.com/thumb_a.jpg'},
            {'id': 'B', 'url': 'https://cdn.example.com/thumb_b.jpg'}
        ],
        'category': 'education'
    }
)

data = response.json()
winner = data['thumbnails'][0]
print(f"Winner: {winner['id']} with CTR score {winner['ctr_score']}")
print(f"Top insight: {winner['insights'][0]}")
```

---

## Subscore Definitions

### Clarity (0-100)
- **What**: Text readability and visual clarity
- **Factors**: Font size, contrast, word count, mobile legibility
- **Good**: 80+
- **Target**: 1-3 words in high-contrast block font

### Subject Prominence (0-100)
- **What**: Face/subject size and positioning
- **Factors**: Subject-to-frame ratio, central positioning, visual weight
- **Good**: 80+
- **Target**: Subject occupies 25-40% of frame, centered on rule-of-thirds

### Contrast Pop (0-100)
- **What**: Color vibrancy and visual impact
- **Factors**: Saturation, complementary colors, brightness contrast
- **Good**: 80+
- **Target**: High saturation (70-90%), strong color contrast

### Emotion (0-100)
- **What**: Emotional expression intensity
- **Factors**: Facial expressions, surprise, excitement, action
- **Good**: 75+
- **Target**: Clear emotion (smile, surprise, shock) or dynamic action

### Hierarchy (0-100)
- **What**: Visual composition and balance
- **Factors**: Focal point clarity, clutter, rule of thirds, negative space
- **Good**: 75+
- **Target**: Single clear focal point, minimal clutter

### Title Match (0-100)
- **What**: Semantic alignment with video title
- **Factors**: CLIP embedding similarity, keyword presence, theme consistency
- **Good**: 80+
- **Target**: Thumbnail visually represents title content

---

## Rate Limits

**Current Limits:**
- 100 requests per minute
- 10 thumbnails max per request
- 30 second timeout per request

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1634567890
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "code": "ERROR_CODE"
}
```

### Common Errors

**400 Bad Request:**
```json
{
  "error": "Invalid request",
  "message": "title and thumbnails are required"
}
```

**429 Too Many Requests:**
```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 100 requests per minute",
  "retry_after": 42
}
```

**500 Internal Server Error:**
```json
{
  "error": "Inference failed",
  "message": "Model prediction error"
}
```

---

## Integration Guide

### 1. Basic Integration

```typescript
class ThumbnailScorer {
  private apiUrl = 'http://localhost:3000/api/v1/score';
  
  async scoreThumbnails(title: string, thumbnails: Array<{id: string, url: string}>) {
    const response = await fetch(this.apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, thumbnails })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return await response.json();
  }
}

// Usage
const scorer = new ThumbnailScorer();
const result = await scorer.scoreThumbnails('My Video Title', [
  { id: 'option1', url: 'https://...' },
  { id: 'option2', url: 'https://...' }
]);

console.log(`Use thumbnail: ${result.winner_id}`);
```

### 2. With Retry Logic

```typescript
async function scoreWithRetry(title: string, thumbnails: any[], maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch('http://localhost:3000/api/v1/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, thumbnails })
      });
      
      if (response.status === 429) {
        // Rate limited - wait and retry
        const retryAfter = parseInt(response.headers.get('Retry-After') || '5');
        await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
        continue;
      }
      
      return await response.json();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

### 3. Batch Processing

```typescript
async function scoreBatch(videos: Array<{title: string, thumbnails: any[]}>) {
  const results = await Promise.all(
    videos.map(async (video) => {
      const result = await fetch('http://localhost:3000/api/v1/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(video)
      }).then(r => r.json());
      
      return {
        title: video.title,
        winner: result.winner_id,
        score: result.thumbnails[0].ctr_score
      };
    })
  );
  
  return results;
}
```

---

## Performance

**Typical Latency:**
- 1 thumbnail: ~50-100ms
- 3 thumbnails: ~150-250ms
- 10 thumbnails: ~500-800ms

**Optimization:**
- Images are cached by URL
- CLIP embeddings are cached
- Model runs on GPU (if available)

---

## Deployment

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

```bash
docker build -t thumbnail-scorer .
docker run -p 3000:3000 thumbnail-scorer
```

### Environment Variables

```bash
# .env.production
API_URL=https://api.thumbnaillab.com
MODEL_VERSION=v3.0.0
ENABLE_GPU=true
MAX_BATCH_SIZE=10
RATE_LIMIT_PER_MINUTE=100
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:3000/api/v1/score
```

### Metrics

Monitor these key metrics:
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Model inference time
- Cache hit rate

---

## Support

For issues or questions:
- Check logs for detailed error messages
- Review rate limits and quotas
- Test with example payload first
- Contact: support@thumbnaillab.com

---

## Changelog

**v1.0.0** (Current)
- Initial release
- Two-stage multi-task model
- 6 interpretable sub-scores
- Visualization overlays
- Rate limiting

**Coming Soon:**
- WebSocket streaming for real-time updates
- Batch upload endpoint
- A/B test tracking integration
- Custom model fine-tuning API
