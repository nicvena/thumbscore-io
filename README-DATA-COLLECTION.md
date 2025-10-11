# YouTube Thumbnail Dataset Collection

This document outlines the process for collecting 100k+ YouTube thumbnails using the YouTube Data API v3 for training our AI model.

## 🎯 Goals

- **Target**: 120k-200k thumbnails across 500-1,500 channels
- **Stratification**: Balanced across 10+ niches (tech, beauty, gaming, etc.)
- **Quality**: High-quality engagement data with proper normalization
- **Compliance**: 100% legal via YouTube Data API v3

## 📊 Data Collection Strategy

### 1. YouTube Data API v3 Integration

```bash
# Get YouTube API key from Google Cloud Console
# Enable YouTube Data API v3
# Add to .env.local:
YOUTUBE_API_KEY=your_api_key_here
```

### 2. Engagement Proxy Calculation

Instead of raw view counts, we use **views_per_hour** to normalize for time:

```typescript
viewsPerHour = viewCount / ((now - publishedAt) / (1000 * 60 * 60))
```

### 3. Channel Baseline Normalization

Calculate z-scores within each channel to reduce creator size bias:

```typescript
zScore = (videoViewsPerHour - channelAvgViewsPerHour) / channelStdDev
```

### 4. Pairwise Training Data

Generate comparisons between videos from the same channel within ±30 days:

```typescript
// Only compare videos from same channel, published within 30 days
// Winner = higher viewsPerHour, Loser = lower viewsPerHour
```

## 🔄 Collection Process

### Phase 1: Channel Discovery (Week 1)
- Search for popular channels in each niche
- Collect channel metadata and recent videos
- Target: 100-150 channels per niche

### Phase 2: Video Collection (Week 2-3)
- Fetch 50-100 videos per channel
- Extract thumbnails, metadata, engagement metrics
- Target: 12k-20k videos per niche

### Phase 3: Data Processing (Week 4)
- Calculate engagement proxies and z-scores
- Generate pairwise comparisons
- Apply quality filters and validation

### Phase 4: Model Preparation (Week 5)
- Split train/validation/test sets (70/15/15)
- Download and preprocess thumbnails
- Extract visual features and metadata

## 🛠 Implementation

### Quick Start

```bash
# 1. Set up environment
cp .env.local.example .env.local
# Add your YouTube API key

# 2. Install dependencies
npm install

# 3. Run data collection
npx ts-node scripts/collect-dataset.ts

# 4. Monitor progress
tail -f data/collected_dataset/logs/collection.log
```

### API Endpoints

```bash
# Test data collection
curl -X POST http://localhost:3001/api/collect-data \
  -H "Content-Type: application/json" \
  -d '{"action": "search_videos", "query": "tech review", "maxResults": 50}'

# Fetch channel videos
curl -X POST http://localhost:3001/api/collect-data \
  -H "Content-Type: application/json" \
  -d '{"action": "fetch_channel", "channelId": "UCBJycsmduvYEL83R_U4JriQ", "maxResults": 100}'

# Generate pairwise data
curl -X POST http://localhost:3001/api/collect-data \
  -H "Content-Type: application/json" \
  -d '{"action": "generate_pairs"}'
```

## 📈 Quality Metrics

### Data Quality Checks
- ✅ Videos have valid thumbnails (downloadable)
- ✅ Engagement metrics are reasonable (views > 100)
- ✅ Publication dates are valid
- ✅ Channel has sufficient video history (10+ videos)

### Niche Distribution
- Target: 12k-20k videos per niche
- Minimum: 8k videos per niche
- Maximum: 25k videos per niche

### Channel Diversity
- Target: 100-150 channels per niche
- Mix of large (1M+ subs) and medium (100k-1M subs) channels
- Geographic diversity (US, UK, CA, AU, etc.)

## 🔒 Legal Compliance

### YouTube Data API v3 Terms
- ✅ No scraping - only official API
- ✅ Respect rate limits (10,000 quota units/day)
- ✅ Store only public metadata
- ✅ No copyrighted content storage

### Data Usage Rights
- ✅ Public thumbnails (fair use for research)
- ✅ Aggregated statistics (no individual privacy concerns)
- ✅ Educational/research purpose
- ✅ No commercial redistribution of raw data

## 📊 Expected Dataset

### Final Structure
```
data/collected_dataset/
├── dataset.json              # Complete dataset
├── videos.json               # All videos with metadata
├── channel_baselines.json    # Channel statistics
├── pairwise_data.json        # Training pairs
├── statistics.json           # Collection statistics
└── thumbnails/              # Downloaded thumbnail images
    ├── tech/
    ├── beauty/
    ├── gaming/
    └── ...
```

### Sample Data Point
```json
{
  "videoId": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "publishedAt": "2009-10-25T06:57:33Z",
  "channelId": "UCuAXFkgsw1L7xaCfnd5JJOw",
  "channelTitle": "Rick Astley",
  "thumbnailUrl": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
  "viewCount": 1234567890,
  "likeCount": 12345678,
  "commentCount": 1234567,
  "viewsPerHour": 1234.56,
  "zScore": 2.34,
  "niche": "entertainment",
  "metadata": {
    "titleLength": 32,
    "hasNumbers": false,
    "hasEmojis": false,
    "hasClickbaitWords": false
  }
}
```

## 🚀 Next Steps

1. **Get YouTube API Key** from Google Cloud Console
2. **Run collection script** with small sample first
3. **Validate data quality** and adjust parameters
4. **Scale up collection** to full dataset
5. **Prepare for model training** with processed data

## 📞 Support

For questions about data collection:
- Check API quotas in Google Cloud Console
- Monitor rate limiting in collection logs
- Validate data quality with statistics.json
- Review YouTube API terms of service
