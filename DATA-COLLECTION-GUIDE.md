# 📊 How to Get Real YouTube Data

Complete guide to collecting 120k-200k YouTube thumbnails for training your model.

---

## Step 1: Get YouTube Data API Key

### 1.1 Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project"
3. Name it: "Thumbnail Lab Data Collection"
4. Click "Create"

### 1.2 Enable YouTube Data API v3

1. In your project, go to **APIs & Services** → **Library**
2. Search for "YouTube Data API v3"
3. Click on it
4. Click "**Enable**"

### 1.3 Create API Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click "**+ Create Credentials**"
3. Select "**API Key**"
4. Copy the API key
5. (Optional) Restrict the key:
   - Click "Edit API key"
   - Under "API restrictions", select "YouTube Data API v3"
   - Click "Save"

### 1.4 Set API Key

```bash
# Add to .env.local
echo "YOUTUBE_API_KEY=YOUR_API_KEY_HERE" >> .env.local

# Or export for Python scripts
export YOUTUBE_API_KEY="YOUR_API_KEY_HERE"
```

---

## Step 2: Understand API Quotas

### Free Tier Limits

- **Daily quota**: 10,000 units per day
- **Cost per request**:
  - `search.list`: 100 units
  - `videos.list`: 1 unit
  - `channels.list`: 1 unit

### Calculate Collection Capacity

```
Daily quota: 10,000 units

Strategy A (Search-heavy):
- 50 search requests × 100 units = 5,000 units
- 5,000 video.list requests × 1 unit = 5,000 units
- Total: ~5,000 videos per day

Strategy B (Efficient):
- 20 channel searches × 100 units = 2,000 units
- Get 50 videos per channel × 1 unit = 50 units per channel
- 20 channels × 50 videos = 1,000 videos per day
- More efficient: uses 2,050 units for 1,000 videos

Recommendation: Use Strategy B
Target: 120k-200k thumbnails
Time needed: 120-200 days (free tier)
OR: Request quota increase (up to 1M units/day)
```

### Request Quota Increase

1. Go to **APIs & Services** → **Quotas**
2. Find "YouTube Data API v3"
3. Click "**Request quota increase**"
4. Explain use case: "Academic research on thumbnail performance"
5. Wait 2-5 business days for approval

**With increased quota (100k units/day):**
- Collect ~10,000 videos per day
- Reach 200k videos in ~20 days

---

## Step 3: Run Data Collection Script

### Option A: Quick Test (100 videos)

```bash
cd python-service/training

# Quick test collection
python3 << EOF
from data_preparation import DataCollector
import os

collector = DataCollector(api_key=os.getenv('YOUTUBE_API_KEY'))

# Get channels from a niche
channels = collector.search_channels_by_niche('education', max_channels=5)
print(f"Found {len(channels)} channels")

# Collect videos
all_videos = []
for channel_id in channels:
    videos = collector.get_channel_videos(channel_id, max_results=20)
    all_videos.extend(videos)
    print(f"Channel {channel_id}: {len(videos)} videos")

print(f"Total collected: {len(all_videos)} videos")
EOF
```

### Option B: Full Collection (120k+ videos)

```bash
# Create collection script
cat > collect_full_dataset.py << 'EOF'
import os
import time
from data_preparation import DataCollector, YouTubeDataIngestion

# Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Target
TARGET_VIDEOS = 120000
CHANNELS_PER_NICHE = 250
VIDEOS_PER_CHANNEL = 50

niches = ['tech', 'gaming', 'education', 'entertainment', 'beauty', 'news']

# Initialize
ingestion = YouTubeDataIngestion(YOUTUBE_API_KEY, SUPABASE_URL, SUPABASE_KEY)
collector = DataCollector(YOUTUBE_API_KEY)

total_videos = 0

for niche in niches:
    print(f"\n{'='*60}")
    print(f"Collecting {niche} niche...")
    print(f"{'='*60}")
    
    # Get channels
    channels = collector.search_channels_by_niche(niche, max_channels=CHANNELS_PER_NICHE)
    print(f"Found {len(channels)} {niche} channels")
    
    for i, channel_id in enumerate(channels):
        try:
            # Get videos from channel
            videos = collector.get_channel_videos(channel_id, max_results=VIDEOS_PER_CHANNEL)
            
            if len(videos) > 0:
                # Store in Supabase
                ingestion.store_in_supabase(videos)
                total_videos += len(videos)
                
                print(f"[{i+1}/{len(channels)}] Channel {channel_id}: {len(videos)} videos (Total: {total_videos})")
            
            # Rate limiting (don't exceed quota)
            time.sleep(1)  # 1 second between channel requests
            
            # Stop if we've reached target
            if total_videos >= TARGET_VIDEOS:
                print(f"\n✅ Target reached: {total_videos} videos collected")
                break
                
        except Exception as e:
            print(f"Error processing {channel_id}: {e}")
            continue
    
    if total_videos >= TARGET_VIDEOS:
        break

print(f"\n{'='*60}")
print(f"Collection Complete: {total_videos} total videos")
print(f"{'='*60}")
EOF

# Run collection
python collect_full_dataset.py
```

**Estimated time:**
- With free quota: 120-200 days
- With increased quota: 12-20 days
- **Tip**: Run as background job

---

## Step 4: Setup Supabase (Database)

### 4.1 Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Sign up / Login
3. Click "**New Project**"
4. Name: "Thumbnail Lab Data"
5. Database password: (choose secure password)
6. Region: (closest to you)
7. Click "**Create new project**"

### 4.2 Get Credentials

1. Go to **Settings** → **API**
2. Copy:
   - **Project URL**: `https://your-project.supabase.co`
   - **anon/public key**: For client-side
   - **service_role key**: For server-side (use this)

### 4.3 Add to Environment

```bash
# Add to .env.local
cat >> .env.local << EOF

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key-here
EOF
```

### 4.4 Create Database Schema

1. In Supabase, go to **SQL Editor**
2. Click "**New Query**"
3. Copy schema from `python-service/training/data_preparation.py`
4. Run the SQL (creates tables: thumbnails, channel_baselines, training_pairs)

**Tables created:**
```sql
✓ thumbnails (stores video metadata + embeddings)
✓ channel_baselines (normalization data)
✓ training_pairs (pairwise comparisons)
✓ model_evaluations (tracking performance)
```

---

## Step 5: Collect Data Efficiently

### Strategy: Maximize Data Quality

**Target Distribution:**
```
Niche            Channels    Videos    Total
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Education        250         50        12,500
Gaming           350         50        17,500
Tech             200         50        10,000
Entertainment    400         50        20,000
Beauty           150         50        7,500
News             150         50        7,500
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL           1,500        50        75,000+

With channel variation (some 30, some 100 videos):
Total target: 120,000-200,000 thumbnails
```

### Batch Collection Script

```python
# collect_batch.py
import os
from data_preparation import DataCollector, YouTubeDataIngestion
import time

# Daily batch (fits in free quota)
DAILY_TARGET = 1000  # videos
BATCH_SIZE = 20      # channels per batch

collector = DataCollector(os.getenv('YOUTUBE_API_KEY'))
ingestion = YouTubeDataIngestion(
    os.getenv('YOUTUBE_API_KEY'),
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

niche = input("Enter niche (tech/gaming/education/entertainment/beauty/news): ")
channels = collector.search_channels_by_niche(niche, max_channels=BATCH_SIZE)

collected = 0
for channel_id in channels:
    videos = collector.get_channel_videos(channel_id, max_results=50)
    if videos:
        ingestion.store_in_supabase(videos)
        collected += len(videos)
        print(f"✓ {channel_id}: {len(videos)} videos (Total: {collected})")
    
    time.sleep(1)  # Rate limiting
    
    if collected >= DAILY_TARGET:
        break

print(f"\n✅ Collected {collected} videos today")
```

**Run daily:**
```bash
# Day 1
python collect_batch.py  # Enter: education

# Day 2
python collect_batch.py  # Enter: gaming

# ... continue for 120-200 days
```

---

## Step 6: Alternative Data Sources (Faster)

### Option 1: YouTube Analytics API (If You Have a Channel)

**For partner channels with access:**

```python
from googleapiclient.discovery import build

youtube_analytics = build('youtubeAnalytics', 'v2', credentials=credentials)

# Get real CTR data
report = youtube_analytics.reports().query(
    ids='channel==MINE',
    startDate='2024-01-01',
    endDate='2024-12-31',
    metrics='views,estimatedMinutesWatched,averageViewDuration,subscribersGained,cardClickRate',
    dimensions='video',
    sort='-views'
).execute()

# Gold-label CTR for training!
```

**Benefits:**
- ✅ Real CTR data (not proxy)
- ✅ Accurate ground truth
- ✅ Better model performance

### Option 2: Existing Datasets

**Public datasets:**
- [YouTube-8M](https://research.google.com/youtube8m/) - 8 million videos
- [YouTube-BB](https://research.google.com/youtube-bb/) - Bounding boxes
- **Note**: May not have thumbnail metadata

### Option 3: Kaggle Datasets

Search Kaggle for:
- "YouTube thumbnails"
- "YouTube video metadata"
- "YouTube statistics"

**Download and adapt** to your format.

---

## Step 7: Verify Data Collection

### Check Supabase

```bash
# Query total videos
curl -X GET "https://your-project.supabase.co/rest/v1/thumbnails?select=count" \
  -H "apikey: YOUR_SUPABASE_KEY" \
  -H "Authorization: Bearer YOUR_SUPABASE_KEY"

# Query by niche
curl -X GET "https://your-project.supabase.co/rest/v1/thumbnails?niche=eq.education&select=count" \
  -H "apikey: YOUR_SUPABASE_KEY"
```

### Sample Data

```sql
-- In Supabase SQL Editor
SELECT 
  niche,
  COUNT(*) as video_count,
  AVG(engagement_proxy) as avg_engagement
FROM thumbnails
GROUP BY niche
ORDER BY video_count DESC;
```

---

## Timeline Estimate

### Scenario 1: Free Tier (10k units/day)

```
Day 1-30:    Tech niche (10k videos)
Day 31-60:   Gaming niche (17k videos)
Day 61-90:   Education niche (12k videos)
Day 91-120:  Entertainment (20k videos)
Day 121-150: Beauty + News (15k videos)
Day 151-180: Additional collection (50k videos)

Total: ~120k videos in 180 days (6 months)
```

### Scenario 2: Increased Quota (100k units/day)

```
Week 1:  All 6 niches × 250 channels = 75k videos
Week 2:  Additional channels = 45k videos
Week 3:  Quality channels + backfill = 80k videos

Total: 200k videos in 3 weeks
```

### Scenario 3: Buy/License Data

**Commercial options:**
- SerpApi YouTube API (paid, higher limits)
- Buy pre-collected dataset from data vendors
- Partner with YouTube channels for Analytics access

**Cost:** ~$500-2,000 for 200k thumbnail dataset

---

## Step 8: Start Small, Scale Up

### Minimum Viable Dataset (MVP)

```bash
# Collect just 5,000 videos to start
Target: 5,000 videos across 100 channels
Time: 5-10 days (free tier)
Use: Train v1 model, test pipeline
```

**Test if it works:**
```bash
cd python-service/training

# Prepare data
python data_preparation.py

# Train model
python pipeline.py --epochs 50 --quick-test

# If AUC > 0.55, you're on the right track!
```

### Then Scale

```
v1: 5k videos   → AUC ~0.55 (proof of concept)
v2: 20k videos  → AUC ~0.62 (usable)
v3: 50k videos  → AUC ~0.68 (good)
v4: 120k videos → AUC ~0.72 (production)
v5: 200k videos → AUC ~0.75+ (excellent)
```

---

## Alternative: Use Simulated Data for Now

**You don't need real data to test everything!**

The app currently works perfectly with simulated ML. You can:

✅ **Build and test UI** (fully functional)  
✅ **Develop features** (Pattern Coach, overlays, etc.)  
✅ **Test API integration**  
✅ **Deploy infrastructure**  
✅ **Show demos to users**  

**Then collect data when you're ready to train the real model.**

---

## Quick Start Commands

### Test YouTube API

```bash
# Test your API key works
curl "https://www.googleapis.com/youtube/v3/search?part=snippet&q=education&type=channel&key=YOUR_API_KEY"
```

### Minimal Collection

```python
# collect_minimal.py
from data_preparation import DataCollector
import os

collector = DataCollector(os.getenv('YOUTUBE_API_KEY'))

# Test with 1 channel
channels = collector.search_channels_by_niche('education', max_channels=1)
videos = collector.get_channel_videos(channels[0], max_results=10)

print(f"✓ Collected {len(videos)} videos")
for v in videos[:3]:
    print(f"  - {v['title'][:50]}... (views: {v['viewCount']:,})")
```

---

## Summary

### To Get Real YouTube Data:

**1. Get API key** (5 minutes)
   - Google Cloud Console → YouTube Data API v3 → Create credentials

**2. Setup Supabase** (10 minutes)
   - Create project → Get credentials → Run schema SQL

**3. Start collecting** (ongoing)
   ```bash
   python collect_batch.py  # Run daily
   ```

**4. Wait for data** (3 weeks - 6 months depending on quota)

**5. Train model** (once you have 20k+ videos)
   ```bash
   python training/pipeline.py
   ```

### Current Recommendation:

**For now:** Use the TypeScript simulation (already works great!)  
**Later:** Collect real data when ready to train production model  
**Optional:** Activate Python service for real CLIP/OCR (without training data)

---

**You can activate Python AI service NOW (real CLIP/OCR/Face detection)**  
**But collecting 120k+ YouTube videos for training takes time**  

**Both paths are ready - choose based on your timeline!** 🚀
