"""
Automated YouTube Thumbnail Library Builder - SCALED UP

This module automatically collects trending YouTube thumbnails every night,
computes CLIP embeddings, and stores them in Supabase for similarity queries.

SCALED FEATURES:
- Fetches 200+ trending videos per niche using YouTube Data API v3 with pagination
- Parallel downloads and CLIP encoding with asyncio + aiohttp
- Batch upserts to Supabase for optimal throughput
- Automatic cleanup of old entries (90+ days)
- Production-safe with comprehensive error handling and logging
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
    "people": 22,         # People & Blogs
    "travel": 19,         # Travel & Events
    "general": 10         # Music (used as general catch-all)
}

# SCALED CONSTANTS
DEFAULT_VIDEOS_PER_NICHE = 200
MAX_CONCURRENT_REQUESTS = 12
BATCH_SIZE = 50  # Supabase batch insert size
REQUEST_TIMEOUT = 30  # seconds


async def fetch_videos_with_pagination(
    niche: str, 
    category_id: int, 
    limit: int = DEFAULT_VIDEOS_PER_NICHE
) -> List[Dict]:
    """
    Fetch trending videos with pagination to get more than 50 results.
    
    Args:
        niche: Niche name
        category_id: YouTube category ID
        limit: Maximum number of videos to fetch
        
    Returns:
        List of video data dictionaries
    """
    videos = []
    page_token = None
    max_results = 50  # YouTube API max per request
    
    logger.info(f"Fetching up to {limit} trending videos for {niche}")
    
    while len(videos) < limit:
        # Calculate how many more we need
        remaining = limit - len(videos)
        current_max = min(max_results, remaining)
        
        # Build API request
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet,statistics,contentDetails",
            "chart": "mostPopular",
            "regionCode": "US",
            "videoCategoryId": str(category_id),
            "maxResults": str(current_max),
            "key": YOUTUBE_API_KEY
        }
        
        if page_token:
            params["pageToken"] = page_token
        
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if "items" not in data:
                logger.warning(f"No items found in API response for {niche}")
                break
            
            # Process videos
            for item in data["items"]:
                if len(videos) >= limit:
                    break
                    
                video_data = {
                    "video_id": item["id"],
                    "title": item["snippet"]["title"],
                    "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"],
                    "published_at": item["snippet"]["publishedAt"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "description": item["snippet"]["description"][:500],  # Truncate
                    "view_count": int(item["statistics"].get("viewCount", 0))
                }
                
                # Calculate views_per_hour
                published = datetime.fromisoformat(item["snippet"]["publishedAt"].replace("Z", "+00:00"))
                hours_ago = (datetime.now(published.tzinfo) - published).total_seconds() / 3600
                video_data["views_per_hour"] = video_data["view_count"] / max(hours_ago, 1)
                
                videos.append(video_data)
            
            # Check for next page
            page_token = data.get("nextPageToken")
            if not page_token:
                logger.info(f"No more pages available for {niche}")
                break
                
            logger.debug(f"Fetched {len(videos)}/{limit} videos for {niche}")
            
        except requests.RequestException as e:
            logger.error(f"API request failed for {niche}: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching videos for {niche}: {e}")
            break
    
    logger.info(f"Fetched {len(videos)} trending videos for {niche}")
    return videos


async def download_and_encode_thumbnail(
    session: aiohttp.ClientSession,
    video_data: Dict,
    niche: str
) -> Optional[Dict]:
    """
    Download thumbnail and compute CLIP embedding for a single video.
    
    Args:
        session: aiohttp session for downloading
        video_data: Video metadata
        niche: Niche category
        
    Returns:
        Enhanced video data with embedding, or None if failed
    """
    try:
        # Download thumbnail
        async with session.get(
            video_data["thumbnail_url"], 
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        ) as response:
            if response.status != 200:
                logger.warning(f"Failed to download thumbnail for {video_data['video_id']}: {response.status}")
                return None
            
            image_data = await response.read()
        
        # Process image and encode
        image = Image.open(io.BytesIO(image_data))
        embedding = clip_encode(image)
        
        if embedding is None:
            logger.warning(f"CLIP encoding failed for {video_data['video_id']}")
            return None
        
        # Add embedding and metadata
        enhanced_data = video_data.copy()
        enhanced_data["embedding"] = embedding.tolist()
        enhanced_data["niche"] = niche
        enhanced_data["collected_at"] = datetime.utcnow().isoformat()
        
        return enhanced_data
        
    except Exception as e:
        logger.error(f"Failed to process thumbnail for {video_data.get('video_id', 'unknown')}: {e}")
        return None


async def process_videos_batch(
    videos: List[Dict], 
    niche: str, 
    semaphore: asyncio.Semaphore
) -> List[Dict]:
    """
    Process a batch of videos with concurrency control.
    
    Args:
        videos: List of video data
        niche: Niche category
        semaphore: Concurrency limiter
        
    Returns:
        List of processed video data with embeddings
    """
    async def process_single(video_data):
        async with semaphore:
            return await download_and_encode_thumbnail(session, video_data, niche)
    
    # Create session with connection limits
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_single(video) for video in videos]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    processed = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
        elif result is not None:
            processed.append(result)
    
    return processed


def batch_upsert_to_supabase(
    client: Client, 
    processed_videos: List[Dict], 
    batch_size: int = BATCH_SIZE
) -> int:
    """
    Upsert processed videos to Supabase in batches for optimal performance.
    
    Args:
        client: Supabase client
        processed_videos: List of processed video data
        batch_size: Number of records per batch
        
    Returns:
        Number of successfully stored videos
    """
    total_stored = 0
    
    # Process in batches
    for i in range(0, len(processed_videos), batch_size):
        batch = processed_videos[i:i + batch_size]
        
        try:
            result = client.table("ref_thumbnails").upsert(
                batch,
                on_conflict="video_id"
            ).execute()
            
            batch_stored = len(result.data) if result.data else 0
            total_stored += batch_stored
            
            logger.debug(f"Stored batch {i//batch_size + 1}: {batch_stored}/{len(batch)} videos")
            
        except Exception as e:
            logger.error(f"Failed to store batch {i//batch_size + 1}: {e}")
            continue
    
    return total_stored


async def collect_niche_thumbnails(
    niche: str, 
    category_id: int, 
    limit: int = DEFAULT_VIDEOS_PER_NICHE
) -> Tuple[int, int]:
    """
    Collect thumbnails for a single niche with parallel processing.
    
    Args:
        niche: Niche name
        category_id: YouTube category ID
        limit: Maximum videos to collect
        
    Returns:
        Tuple of (videos_fetched, thumbnails_stored)
    """
    logger.info(f"Processing niche: {niche} (category {category_id})")
    
    # Step 1: Fetch videos with pagination
    videos = await fetch_videos_with_pagination(niche, category_id, limit)
    
    if not videos:
        logger.warning(f"No videos fetched for {niche}")
        return 0, 0
    
    # Step 2: Process videos in parallel with concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    processed_videos = await process_videos_batch(videos, niche, semaphore)
    
    logger.info(f"Processed {len(processed_videos)}/{len(videos)} thumbnails for {niche}")
    
    # Step 3: Store in Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        stored_count = batch_upsert_to_supabase(client, processed_videos)
        logger.info(f"Stored {stored_count}/{len(processed_videos)} thumbnails for {niche}")
    else:
        logger.error("Supabase credentials not configured")
        stored_count = 0
    
    return len(videos), stored_count


def cleanup_old_thumbnails(client: Client, days_old: int = 90) -> int:
    """
    Clean up thumbnails older than specified days.
    
    Args:
        client: Supabase client
        days_old: Age threshold in days
        
    Returns:
        Number of deleted records
    """
    cutoff_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
    
    try:
        result = client.table("ref_thumbnails").delete().lt(
            "collected_at", cutoff_date
        ).execute()
        
        deleted_count = len(result.data) if result.data else 0
        logger.info(f"Cleaned up {deleted_count} old thumbnails (older than {days_old} days)")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup old thumbnails: {e}")
        return 0


async def update_reference_library(
    limit_per_niche: int = DEFAULT_VIDEOS_PER_NICHE
) -> Dict[str, any]:
    """
    Main function to update the reference thumbnail library.
    
    Args:
        limit_per_niche: Maximum videos to collect per niche
        
    Returns:
        Dictionary with collection statistics
    """
    start_time = time.time()
    
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY environment variable is required")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
    
    logger.info("[Thumbscore] Updating reference thumbnail library...")
    logger.info(f"Target: {limit_per_niche} videos per niche ({len(NICHES)} niches)")
    
    # Initialize Supabase client
    # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Collect thumbnails for all niches in parallel
    tasks = []
    for niche, category_id in NICHES.items():
        task = collect_niche_thumbnails(niche, category_id, limit_per_niche)
        tasks.append(task)
    
    # Wait for all niches to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    total_videos_fetched = 0
    total_thumbnails_stored = 0
    niche_stats = {}
    
    for i, (niche, result) in enumerate(zip(NICHES.keys(), results)):
        if isinstance(result, Exception):
            logger.error(f"Failed to collect thumbnails for {niche}: {result}")
            niche_stats[niche] = {"fetched": 0, "stored": 0}
        else:
            fetched, stored = result
            total_videos_fetched += fetched
            total_thumbnails_stored += stored
            niche_stats[niche] = {"fetched": fetched, "stored": stored}
    
    # Cleanup old thumbnails
    old_deleted = cleanup_old_thumbnails(client)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Log comprehensive summary
    logger.info("=" * 60)
    logger.info("📊 COLLECTION SUMMARY")
    logger.info("=" * 60)
    
    for niche, stats in niche_stats.items():
        logger.info(f"  {niche:12s}: {stats['fetched']:3d} fetched | {stats['stored']:3d} stored")
    
    logger.info("-" * 60)
    logger.info(f"  TOTAL       : {total_videos_fetched:3d} fetched | {total_thumbnails_stored:3d} stored")
    logger.info(f"  Processing  : {processing_time:.1f}s ({total_thumbnails_stored/processing_time:.1f} thumbnails/sec)")
    logger.info(f"  Cleanup     : {old_deleted} old thumbnails removed")
    logger.info(f"  Status      : {'✅ SUCCESS' if total_thumbnails_stored > 0 else '❌ FAILED'}")
    logger.info("=" * 60)
    
    return {
        "total_videos_fetched": total_videos_fetched,
        "total_thumbnails_stored": total_thumbnails_stored,
        "niches_processed": len(NICHES),
        "old_thumbnails_deleted": old_deleted,
        "processing_time_seconds": processing_time,
        "niche_stats": niche_stats,
        "throughput_thumbnails_per_sec": total_thumbnails_stored / processing_time if processing_time > 0 else 0
    }


def update_reference_library_sync(limit_per_niche: int = DEFAULT_VIDEOS_PER_NICHE) -> Dict[str, any]:
    """
    Synchronous wrapper for the async update function.
    
    Args:
        limit_per_niche: Maximum videos to collect per niche
        
    Returns:
        Dictionary with collection statistics
    """
    return asyncio.run(update_reference_library(limit_per_niche))


if __name__ == "__main__":
    # Test the collection process
    try:
        stats = update_reference_library_sync()
        print(f"Collection completed: {stats}")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise