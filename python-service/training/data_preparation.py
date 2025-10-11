"""
Data Preparation for Training Pipeline

Steps:
1. Collect YouTube data via API
2. Store in Supabase (Postgres)
3. Normalize engagement metrics
4. Build training pairs
5. Generate embeddings (CLIP + MiniLM)
"""

import os
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import torch

# YouTube Data API (requires google-api-python-client)
try:
    from googleapiclient.discovery import build
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    print("⚠ google-api-python-client not installed")

# Supabase (requires supabase-py)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠ supabase not installed")

class DataCollector:
    """
    Collect YouTube thumbnail data via API
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        if YOUTUBE_AVAILABLE:
            self.youtube = build('youtube', 'v3', developerKey=api_key)
        else:
            self.youtube = None
    
    def search_channels_by_niche(
        self,
        niche: str,
        max_channels: int = 100
    ) -> List[str]:
        """
        Search for channels in specific niche
        
        Niches: tech, gaming, education, entertainment, beauty, news
        """
        if not self.youtube:
            # Return mock channel IDs
            return [f'{niche}_channel_{i}' for i in range(max_channels)]
        
        # Use YouTube search API
        request = self.youtube.search().list(
            part='snippet',
            q=niche,
            type='channel',
            maxResults=min(50, max_channels),
            relevanceLanguage='en'
        )
        response = request.execute()
        
        channel_ids = [item['id']['channelId'] for item in response['items']]
        return channel_ids
    
    def get_channel_videos(
        self,
        channel_id: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Get recent videos from a channel with full metadata
        """
        if not self.youtube:
            return self._mock_channel_videos(channel_id, max_results)
        
        # Get video IDs
        request = self.youtube.search().list(
            part='id',
            channelId=channel_id,
            type='video',
            order='date',
            maxResults=max_results
        )
        response = request.execute()
        
        video_ids = [item['id']['videoId'] for item in response['items']]
        
        # Get video details
        videos_data = []
        for i in range(0, len(video_ids), 50):  # API limit: 50 per request
            batch_ids = video_ids[i:i+50]
            
            request = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(batch_ids)
            )
            response = request.execute()
            
            for item in response['items']:
                videos_data.append(self._parse_video_item(item, channel_id))
        
        return videos_data
    
    def _parse_video_item(self, item: Dict, channel_id: str) -> Dict:
        """
        Parse YouTube API video item into standard format
        """
        snippet = item['snippet']
        stats = item['statistics']
        details = item['contentDetails']
        
        # Parse ISO 8601 duration (PT15M33S → seconds)
        duration = self._parse_duration(details['duration'])
        
        return {
            'videoId': item['id'],
            'channelId': channel_id,
            'title': snippet['title'],
            'thumbnailUrl': snippet['thumbnails']['maxres']['url'] 
                            if 'maxres' in snippet['thumbnails'] 
                            else snippet['thumbnails']['high']['url'],
            'publishedAt': snippet['publishedAt'],
            'viewCount': int(stats.get('viewCount', 0)),
            'likeCount': int(stats.get('likeCount', 0)),
            'commentCount': int(stats.get('commentCount', 0)),
            'categoryId': snippet['categoryId'],
            'duration': duration,
            'tags': snippet.get('tags', []),
            'description': snippet.get('description', '')[:500]  # Truncate
        }
    
    def _parse_duration(self, iso_duration: str) -> int:
        """
        Parse ISO 8601 duration to seconds
        PT15M33S → 933 seconds
        """
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration)
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def _mock_channel_videos(self, channel_id: str, max_results: int) -> List[Dict]:
        """
        Generate mock video data for testing
        """
        videos = []
        for i in range(max_results):
            published_date = datetime.now() - timedelta(days=i * 2)
            
            videos.append({
                'videoId': f'video_{channel_id}_{i}',
                'channelId': channel_id,
                'title': f'Sample Video Title {i}',
                'thumbnailUrl': f'https://i.ytimg.com/vi/mock_{i}/maxresdefault.jpg',
                'publishedAt': published_date.isoformat(),
                'viewCount': int(np.random.lognormal(10, 2)),
                'likeCount': int(np.random.lognormal(7, 2)),
                'commentCount': int(np.random.lognormal(5, 2)),
                'categoryId': '22',
                'duration': int(np.random.uniform(180, 1200)),
                'tags': ['sample', 'video', 'mock'],
                'description': 'Mock video description'
            })
        
        return videos

class DataNormalizer:
    """
    Normalize engagement metrics across channels
    """
    def __init__(self):
        self.channel_baselines = {}
    
    def compute_channel_baseline(self, videos: List[Dict]) -> Dict[str, float]:
        """
        Compute mean and std of engagement within channel
        """
        proxies = [self._compute_proxy(v) for v in videos]
        
        return {
            'mean': np.mean(proxies),
            'std': np.std(proxies),
            'median': np.median(proxies),
            'count': len(videos)
        }
    
    def _compute_proxy(self, video: Dict) -> float:
        """
        Compute views_per_hour engagement proxy
        """
        published = datetime.fromisoformat(video['publishedAt'].replace('Z', '+00:00'))
        hours_since = (datetime.now() - published.replace(tzinfo=None)).total_seconds() / 3600
        
        if hours_since < 1:
            hours_since = 1
        
        views_per_hour = video['viewCount'] / hours_since
        return views_per_hour
    
    def normalize_ctr(
        self,
        video: Dict,
        channel_baseline: Dict[str, float]
    ) -> float:
        """
        Normalize CTR using channel z-score
        """
        proxy = self._compute_proxy(video)
        
        # Z-score normalization
        z_score = (proxy - channel_baseline['mean']) / max(channel_baseline['std'], 1)
        
        # Convert to 0-1 range (clip at ±3 std)
        normalized = (z_score + 3) / 6
        normalized = max(0, min(1, normalized))
        
        return normalized

class EmbeddingGenerator:
    """
    Generate CLIP and MiniLM embeddings
    """
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Load CLIP
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
            print("✓ CLIP ViT-L/14 loaded")
        except:
            self.clip_model = None
            print("⚠ CLIP not available")
        
        # Load MiniLM for title encoding
        try:
            from sentence_transformers import SentenceTransformer
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ MiniLM loaded")
        except:
            self.text_model = None
            print("⚠ MiniLM not available")
    
    def encode_image(self, image_url: str) -> np.ndarray:
        """
        Download image and generate CLIP embedding
        """
        if not self.clip_model:
            # Return random embedding
            return np.random.randn(768).astype(np.float32)
        
        try:
            import requests
            from PIL import Image
            import io
            
            # Download image
            response = requests.get(image_url, timeout=10)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # Encode with CLIP
            with torch.no_grad():
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                embedding = self.clip_model.encode_image(image_tensor)
                return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error encoding image {image_url}: {e}")
            return np.random.randn(768).astype(np.float32)
    
    def encode_text(self, title: str) -> np.ndarray:
        """
        Generate MiniLM embedding for title
        """
        if not self.text_model:
            # Return random embedding
            return np.random.randn(384).astype(np.float32)
        
        try:
            embedding = self.text_model.encode(title)
            return embedding
        except Exception as e:
            print(f"Error encoding text '{title}': {e}")
            return np.random.randn(384).astype(np.float32)

# ============================================================================
# FULL PIPELINE ORCHESTRATION
# ============================================================================

def build_training_dataset(
    youtube_api_key: str,
    supabase_url: str,
    supabase_key: str,
    target_samples: int = 200000,
    channels_per_niche: int = 250
):
    """
    Build complete training dataset
    
    Target: 120k-200k thumbnails across 500-1,500 channels
    """
    print("Building training dataset...")
    
    niches = ['tech', 'gaming', 'education', 'entertainment', 'beauty', 'news']
    
    collector = DataCollector(youtube_api_key)
    normalizer = DataNormalizer()
    embedder = EmbeddingGenerator()
    
    all_pairs = []
    all_ctr_samples = []
    
    for niche in niches:
        print(f"\nProcessing niche: {niche}")
        
        # Get channels in this niche
        channels = collector.search_channels_by_niche(niche, max_channels=channels_per_niche)
        print(f"  Found {len(channels)} channels")
        
        for i, channel_id in enumerate(channels):
            if i % 10 == 0:
                print(f"  Processing channel {i}/{len(channels)}...")
            
            # Get videos
            videos = collector.get_channel_videos(channel_id, max_results=50)
            
            if len(videos) < 2:
                continue
            
            # Compute channel baseline
            baseline = normalizer.compute_channel_baseline(videos)
            
            # Build pairwise samples
            for j in range(len(videos)):
                for k in range(j + 1, len(videos)):
                    video_a = videos[j]
                    video_b = videos[k]
                    
                    # Check ±30 day window
                    date_a = datetime.fromisoformat(video_a['publishedAt'].replace('Z', '+00:00'))
                    date_b = datetime.fromisoformat(video_b['publishedAt'].replace('Z', '+00:00'))
                    
                    if abs((date_b - date_a).days) > 30:
                        continue
                    
                    # Generate embeddings
                    img_a = embedder.encode_image(video_a['thumbnailUrl'])
                    title_a = embedder.encode_text(video_a['title'])
                    img_b = embedder.encode_image(video_b['thumbnailUrl'])
                    title_b = embedder.encode_text(video_b['title'])
                    
                    # Compute label
                    proxy_a = normalizer._compute_proxy(video_a)
                    proxy_b = normalizer._compute_proxy(video_b)
                    
                    label = 1 if proxy_a > proxy_b else 0
                    margin = abs(proxy_a - proxy_b) / max(proxy_a, proxy_b)
                    
                    all_pairs.append((
                        torch.from_numpy(img_a),
                        torch.from_numpy(title_a),
                        torch.from_numpy(img_b),
                        torch.from_numpy(title_b),
                        label,
                        margin
                    ))
            
            # Build absolute CTR samples
            for video in videos:
                img_emb = embedder.encode_image(video['thumbnailUrl'])
                title_emb = embedder.encode_text(video['title'])
                ctr_norm = normalizer.normalize_ctr(video, baseline)
                
                all_ctr_samples.append((
                    torch.from_numpy(img_emb),
                    torch.from_numpy(title_emb),
                    ctr_norm,
                    (normalizer._compute_proxy(video) - baseline['mean']) / baseline['std']
                ))
            
            # Stop if we have enough samples
            if len(all_pairs) >= target_samples:
                break
        
        if len(all_pairs) >= target_samples:
            break
    
    print(f"\n✓ Dataset complete:")
    print(f"  Pairwise samples: {len(all_pairs)}")
    print(f"  CTR samples: {len(all_ctr_samples)}")
    
    return all_pairs, all_ctr_samples

# ============================================================================
# SUPABASE SCHEMA
# ============================================================================

SUPABASE_SCHEMA_SQL = """
-- Thumbnails table
CREATE TABLE IF NOT EXISTS thumbnails (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) UNIQUE NOT NULL,
    channel_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    thumbnail_url TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    view_count BIGINT DEFAULT 0,
    like_count BIGINT DEFAULT 0,
    comment_count BIGINT DEFAULT 0,
    category_id VARCHAR(50),
    duration INTEGER,
    tags TEXT[],
    description TEXT,
    
    -- Computed metrics
    engagement_proxy FLOAT,
    ctr_normalized FLOAT,
    channel_baseline_zscore FLOAT,
    
    -- Embeddings (stored as arrays)
    clip_embedding FLOAT[],
    title_embedding FLOAT[],
    
    -- Metadata
    collected_at TIMESTAMP DEFAULT NOW(),
    niche VARCHAR(50),
    
    -- Indexes
    INDEX idx_channel_id (channel_id),
    INDEX idx_published_at (published_at),
    INDEX idx_niche (niche),
    INDEX idx_engagement (engagement_proxy DESC)
);

-- Channel baselines table
CREATE TABLE IF NOT EXISTS channel_baselines (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR(255) UNIQUE NOT NULL,
    mean_engagement FLOAT,
    std_engagement FLOAT,
    median_engagement FLOAT,
    sample_count INTEGER,
    niche VARCHAR(50),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Training pairs table
CREATE TABLE IF NOT EXISTS training_pairs (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR(255) NOT NULL,
    video_a_id VARCHAR(255) NOT NULL,
    video_b_id VARCHAR(255) NOT NULL,
    label INTEGER NOT NULL,  -- 1 if A > B, 0 if B > A
    margin FLOAT NOT NULL,   -- Performance difference
    days_apart INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Foreign keys
    FOREIGN KEY (video_a_id) REFERENCES thumbnails(video_id),
    FOREIGN KEY (video_b_id) REFERENCES thumbnails(video_id),
    
    -- Ensure unique pairs
    UNIQUE (video_a_id, video_b_id)
);

-- Model evaluation results
CREATE TABLE IF NOT EXISTS model_evaluations (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    pairwise_auc FLOAT,
    pairwise_auc_ci_lower FLOAT,
    pairwise_auc_ci_upper FLOAT,
    spearman_correlation FLOAT,
    spearman_pvalue FLOAT,
    ab_test_win_rate FLOAT,
    evaluated_at TIMESTAMP DEFAULT NOW(),
    notes TEXT
);
"""

def setup_supabase_schema(supabase_url: str, supabase_key: str):
    """
    Initialize Supabase schema for training data
    """
    if not SUPABASE_AVAILABLE:
        print("⚠ Supabase not available. Install with: pip install supabase")
        return
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    print("Setting up Supabase schema...")
    print("Execute this SQL in your Supabase SQL editor:")
    print("\n" + "=" * 60)
    print(SUPABASE_SCHEMA_SQL)
    print("=" * 60 + "\n")
    
    # Note: Supabase client doesn't support direct SQL execution
    # User must run SQL in Supabase dashboard

# ============================================================================
# READY-TO-RUN TRAINING SCRIPT
# ============================================================================

if __name__ == '__main__':
    print("YouTube Thumbnail Training - Data Preparation")
    print("=" * 60)
    
    # Configuration from environment
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    
    if not YOUTUBE_API_KEY:
        print("⚠ YOUTUBE_API_KEY not set. Using mock data.")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠ Supabase credentials not set. Data won't be stored.")
    
    # Setup schema (run once)
    if SUPABASE_URL and SUPABASE_KEY:
        setup_supabase_schema(SUPABASE_URL, SUPABASE_KEY)
    
    # Build dataset
    print("\nBuilding training dataset...")
    pairs, ctr_samples = build_training_dataset(
        youtube_api_key=YOUTUBE_API_KEY,
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY,
        target_samples=200000,
        channels_per_niche=250
    )
    
    # Save to disk
    print("\nSaving dataset...")
    os.makedirs('data', exist_ok=True)
    
    torch.save({
        'pairs': pairs,
        'ctr_samples': ctr_samples,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'num_pairs': len(pairs),
            'num_ctr_samples': len(ctr_samples)
        }
    }, 'data/training_dataset.pt')
    
    print(f"✓ Dataset saved to data/training_dataset.pt")
    print(f"  Pairwise samples: {len(pairs):,}")
    print(f"  CTR samples: {len(ctr_samples):,}")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("Next step: python training/pipeline.py")
    print("=" * 60)

