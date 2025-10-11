"""
Automated YouTube Thumbnail Library Builder

This module automatically collects trending YouTube thumbnails every night,
computes CLIP embeddings, and stores them in Supabase for similarity queries.

Features:
- Fetches trending videos from 5 niches using YouTube Data API v3
- Downloads thumbnails and computes CLIP embeddings
- Stores in Supabase with metadata (views_per_hour, title, etc.)
- Automatically cleans up old entries (90+ days)
- Production-safe with proper error handling and logging
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import aiohttp
import numpy as np
from supabase import create_client, Client
import requests
from PIL import Image
import io

from app.features import clip_encode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YouTube Data API configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Niche categories with YouTube category IDs
NICHES = {
    "tech": 28,           # Science & Technology
    "gaming": 20,         # Gaming
    "education": 27,      # Education
    "entertainment": 24,  # Entertainment
    "people": 22          # People & Blogs
}

# Constants
VIDEOS_PER_NICHE = 30
MAX_THUMBNAIL_AGE_DAYS = 90
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


class ThumbnailCollector:
    """Handles automated collection of trending YouTube thumbnails."""
    
    def __init__(self):
        """Initialize the collector with API credentials."""
        if not YOUTUBE_API_KEY:
            raise ValueError("YOUTUBE_API_KEY environment variable is required")
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def fetch_trending_videos(self, niche: str, category_id: int) -> List[Dict]:
        """
        Fetch trending videos for a specific niche.
        
        Args:
            niche: Human-readable niche name
            category_id: YouTube category ID
            
        Returns:
            List of video metadata dictionaries
        """
        url = f"{YOUTUBE_API_BASE}/videos"
        params = {
            "part": "snippet,statistics",
            "chart": "mostPopular",
            "regionCode": "US",  # Can be made configurable
            "maxResults": VIDEOS_PER_NICHE,
            "videoCategoryId": category_id,
            "key": YOUTUBE_API_KEY
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch videos for {niche}: {response.status}")
                    return []
                
                data = await response.json()
                videos = []
                
                for item in data.get("items", []):
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    
                    # Calculate views per hour
                    published_at = snippet.get("publishedAt", "")
                    view_count = int(stats.get("viewCount", 0))
                    views_per_hour = self._calculate_views_per_hour(published_at, view_count)
                    
                    video_data = {
                        "video_id": item.get("id"),
                        "title": snippet.get("title", ""),
                        "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                        "published_at": published_at,
                        "view_count": view_count,
                        "views_per_hour": views_per_hour,
                        "channel_title": snippet.get("channelTitle", ""),
                        "description": snippet.get("description", "")[:500]  # Truncate for storage
                    }
                    videos.append(video_data)
                
                logger.info(f"Fetched {len(videos)} trending videos for {niche}")
                return videos
                
        except Exception as e:
            logger.error(f"Error fetching videos for {niche}: {e}")
            return []

    def _calculate_views_per_hour(self, published_at: str, view_count: int) -> float:
        """
        Calculate views per hour since publication.
        
        Args:
            published_at: ISO 8601 timestamp
            view_count: Total view count
            
        Returns:
            Views per hour (float)
        """
        try:
            pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            hours_since_pub = (datetime.now(pub_date.tzinfo) - pub_date).total_seconds() / 3600
            
            if hours_since_pub <= 0:
                return float(view_count)  # Just published
                
            return float(view_count) / hours_since_pub
            
        except Exception as e:
            logger.warning(f"Error calculating views per hour: {e}")
            return 0.0

    async def download_and_encode_thumbnail(self, thumbnail_url: str) -> Optional[np.ndarray]:
        """
        Download thumbnail and compute CLIP embedding.
        
        Args:
            thumbnail_url: URL of the thumbnail image
            
        Returns:
            CLIP embedding as numpy array, or None if failed
        """
        try:
            # Download image
            async with self.session.get(thumbnail_url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download thumbnail: {response.status}")
                    return None
                
                image_data = await response.read()
            
            # Load image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Compute CLIP embedding
            embedding = clip_encode(image)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Error processing thumbnail {thumbnail_url}: {e}")
            return None

    async def store_thumbnails(self, niche: str, videos: List[Dict]) -> int:
        """
        Store video thumbnails with embeddings in Supabase.
        
        Args:
            niche: Niche category
            videos: List of video metadata
            
        Returns:
            Number of successfully stored thumbnails
        """
        stored_count = 0
        
        for video in videos:
            try:
                # Download and encode thumbnail
                embedding = await self.download_and_encode_thumbnail(video["thumbnail_url"])
                
                if embedding is None:
                    logger.warning(f"Skipping video {video['video_id']} - failed to process thumbnail")
                    continue
                
                # Prepare data for Supabase
                thumbnail_data = {
                    "video_id": video["video_id"],
                    "niche": niche,
                    "title": video["title"],
                    "thumbnail_url": video["thumbnail_url"],
                    "views_per_hour": video["views_per_hour"],
                    "view_count": video["view_count"],
                    "published_at": video["published_at"],
                    "channel_title": video["channel_title"],
                    "description": video["description"],
                    "embedding": embedding.tolist(),  # Convert numpy array to list
                    "collected_at": datetime.utcnow().isoformat()
                }
                
                # Upsert into Supabase
                result = self.supabase.table("ref_thumbnails").upsert(
                    thumbnail_data,
                    on_conflict="video_id"
                ).execute()
                
                if result.data:
                    stored_count += 1
                    logger.debug(f"Stored thumbnail for video: {video['title'][:50]}...")
                else:
                    logger.warning(f"Failed to store thumbnail for video: {video['video_id']}")
                    
            except Exception as e:
                logger.error(f"Error storing thumbnail for video {video.get('video_id', 'unknown')}: {e}")
                continue
        
        return stored_count

    async def cleanup_old_thumbnails(self) -> int:
        """
        Remove thumbnails older than MAX_THUMBNAIL_AGE_DAYS.
        
        Returns:
            Number of deleted thumbnails
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=MAX_THUMBNAIL_AGE_DAYS)
            cutoff_iso = cutoff_date.isoformat()
            
            # Delete old entries
            result = self.supabase.table("ref_thumbnails").delete().lt(
                "collected_at", cutoff_iso
            ).execute()
            
            deleted_count = len(result.data) if result.data else 0
            logger.info(f"Cleaned up {deleted_count} old thumbnails (older than {MAX_THUMBNAIL_AGE_DAYS} days)")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old thumbnails: {e}")
            return 0

    async def update_reference_library(self) -> Dict[str, int]:
        """
        Main function to update the entire reference thumbnail library.
        
        Returns:
            Dictionary with collection statistics
        """
        logger.info("Updating reference thumbnail library...")
        start_time = time.time()
        
        stats = {
            "total_videos_fetched": 0,
            "total_thumbnails_stored": 0,
            "niches_processed": 0,
            "old_thumbnails_deleted": 0,
            "processing_time_seconds": 0
        }
        
        try:
            async with self as collector:
                # Process each niche
                for niche, category_id in NICHES.items():
                    logger.info(f"Processing niche: {niche} (category {category_id})")
                    
                    # Fetch trending videos
                    videos = await collector.fetch_trending_videos(niche, category_id)
                    stats["total_videos_fetched"] += len(videos)
                    
                    if videos:
                        # Store thumbnails with embeddings
                        stored_count = await collector.store_thumbnails(niche, videos)
                        stats["total_thumbnails_stored"] += stored_count
                        stats["niches_processed"] += 1
                        
                        logger.info(f"Stored {stored_count}/{len(videos)} thumbnails for {niche}")
                
                # Cleanup old thumbnails
                deleted_count = await collector.cleanup_old_thumbnails()
                stats["old_thumbnails_deleted"] = deleted_count
            
            # Calculate processing time
            stats["processing_time_seconds"] = round(time.time() - start_time, 2)
            
            logger.info(f"Reference library refreshed. Stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error updating reference library: {e}")
            raise


# Convenience function for external use
async def update_reference_library() -> Dict[str, int]:
    """
    Public function to update the reference thumbnail library.
    
    Returns:
        Dictionary with collection statistics
    """
    collector = ThumbnailCollector()
    return await collector.update_reference_library()


# Synchronous wrapper for scheduler compatibility
def update_reference_library_sync() -> Dict[str, int]:
    """
    Synchronous wrapper for APScheduler compatibility.
    
    Returns:
        Dictionary with collection statistics
    """
    return asyncio.run(update_reference_library())


if __name__ == "__main__":
    # Test the collector
    async def test_collector():
        stats = await update_reference_library()
        print(f"Collection completed: {stats}")
    
    asyncio.run(test_collector())
