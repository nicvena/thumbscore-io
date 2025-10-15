"""
YouTube Intelligence Brain - Data Collector
Collects trending thumbnails and metadata daily to build our knowledge base
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class VideoData:
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    thumbnail_url: str
    published_at: datetime
    view_count: int
    like_count: int
    comment_count: int
    duration: str
    category_id: str
    tags: List[str]
    description: str
    views_per_hour: float
    engagement_rate: float

class YouTubeDataCollector:
    """
    Collects trending YouTube data to build our intelligence base
    """
    
    def __init__(self, api_key: str, supabase_client):
        self.api_key = api_key
        self.supabase = supabase_client
        self.session = None
        
        # Niche categories with their YouTube category IDs
        self.niche_categories = {
            "tech": "28",           # Science & Technology
            "gaming": "20",         # Gaming
            "education": "27",      # Education
            "entertainment": "24",  # Entertainment
            "people": "22",         # People & Blogs
            "business": "25",       # News & Politics (closest to business/finance)
            "music": "10",          # Music
            "sports": "17",         # Sports
            "news": "25",          # News & Politics
            "comedy": "23",        # Comedy
            "howto": "26"          # Howto & Style
        }
    
    async def collect_trending_data(self, max_videos_per_niche: int = 200) -> Dict[str, List[VideoData]]:
        """
        Collect trending videos across all niches
        """
        logger.info(f"[BRAIN] Starting data collection for {len(self.niche_categories)} niches")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            all_data = {}
            
            for niche, category_id in self.niche_categories.items():
                try:
                    logger.info(f"[BRAIN] Collecting {niche} videos...")
                    niche_data = await self._collect_niche_videos(niche, category_id, max_videos_per_niche)
                    all_data[niche] = niche_data
                    logger.info(f"[BRAIN] Collected {len(niche_data)} videos for {niche}")
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"[BRAIN] Failed to collect {niche} videos: {e}")
                    all_data[niche] = []
            
            return all_data
    
    async def _collect_niche_videos(self, niche: str, category_id: str, max_videos: int) -> List[VideoData]:
        """
        Collect videos for a specific niche using multiple strategies
        """
        videos = []
        
        # Strategy 1: Trending videos
        trending_videos = await self._get_trending_videos(category_id, max_videos // 2)
        videos.extend(trending_videos)
        
        # Strategy 2: High-performing videos from popular channels
        popular_videos = await self._get_popular_channel_videos(niche, max_videos // 2)
        videos.extend(popular_videos)
        
        # Remove duplicates and sort by performance
        unique_videos = self._deduplicate_videos(videos)
        return sorted(unique_videos, key=lambda v: v.views_per_hour, reverse=True)[:max_videos]
    
    async def _get_trending_videos(self, category_id: str, limit: int) -> List[VideoData]:
        """
        Get trending videos for a category
        """
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet,statistics,contentDetails",
            "chart": "mostPopular",
            "regionCode": "US",
            "maxResults": min(50, limit),
            "videoCategoryId": category_id,
            "key": self.api_key
        }
        
        videos = []
        page_token = None
        
        while len(videos) < limit and page_token != "STOP":
            if page_token:
                params["pageToken"] = page_token
            
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get("items", []):
                            video_data = self._parse_video_data(item)
                            if video_data:
                                videos.append(video_data)
                        
                        page_token = data.get("nextPageToken")
                        if not page_token:
                            page_token = "STOP"
                    else:
                        logger.warning(f"[BRAIN] API error: {response.status}")
                        break
                        
            except Exception as e:
                logger.error(f"[BRAIN] Error fetching trending videos: {e}")
                break
        
        return videos
    
    async def _get_popular_channel_videos(self, niche: str, limit: int) -> List[VideoData]:
        """
        Get recent videos from popular channels in the niche
        """
        # Get popular channels for the niche
        channels = await self._get_popular_channels(niche)
        
        videos = []
        for channel_id in channels[:10]:  # Top 10 channels
            channel_videos = await self._get_channel_recent_videos(channel_id, limit // 10)
            videos.extend(channel_videos)
        
        return videos
    
    async def _get_popular_channels(self, niche: str) -> List[str]:
        """
        Get popular channel IDs for a niche
        """
        # This would search for popular channels in the niche
        # For now, return some known popular channels per niche
        popular_channels = {
            "tech": ["UCBJycsmduvYEL83R_U4JriQ", "UCXuqSBlHAE6Xw-yeJA0Tunw"],  # Marques, Linus
            "gaming": ["UCBJycsmduvYEL83R_U4JriQ", "UCBJycsmduvYEL83R_U4JriQ"],  # Popular gaming channels
            "education": ["UCBJycsmduvYEL83R_U4JriQ", "UCBJycsmduvYEL83R_U4JriQ"],  # Educational channels
            "entertainment": ["UCBJycsmduvYEL83R_U4JriQ", "UCBJycsmduvYEL83R_U4JriQ"],  # Entertainment channels
        }
        
        return popular_channels.get(niche, [])
    
    async def _get_channel_recent_videos(self, channel_id: str, limit: int) -> List[VideoData]:
        """
        Get recent videos from a specific channel
        """
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "maxResults": min(50, limit),
            "key": self.api_key
        }
        
        videos = []
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Get detailed video info
                    video_ids = [item["id"]["videoId"] for item in data.get("items", [])]
                    detailed_videos = await self._get_video_details(video_ids)
                    videos.extend(detailed_videos)
                    
        except Exception as e:
            logger.error(f"[BRAIN] Error fetching channel videos: {e}")
        
        return videos
    
    async def _get_video_details(self, video_ids: List[str]) -> List[VideoData]:
        """
        Get detailed information for a list of video IDs
        """
        if not video_ids:
            return []
        
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": self.api_key
        }
        
        videos = []
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data.get("items", []):
                        video_data = self._parse_video_data(item)
                        if video_data:
                            videos.append(video_data)
                            
        except Exception as e:
            logger.error(f"[BRAIN] Error fetching video details: {e}")
        
        return videos
    
    def _parse_video_data(self, item: Dict[str, Any]) -> Optional[VideoData]:
        """
        Parse YouTube API response into VideoData object
        """
        try:
            snippet = item["snippet"]
            statistics = item["statistics"]
            content_details = item["contentDetails"]
            
            # Calculate views per hour
            published_at = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
            hours_since_published = max(1, (datetime.now(published_at.tzinfo) - published_at).total_seconds() / 3600)
            
            view_count = int(statistics.get("viewCount", 0))
            like_count = int(statistics.get("likeCount", 0))
            comment_count = int(statistics.get("commentCount", 0))
            
            views_per_hour = view_count / hours_since_published
            engagement_rate = (like_count + comment_count) / max(view_count, 1) if view_count > 0 else 0
            
            return VideoData(
                video_id=item["id"],
                title=snippet["title"],
                channel_id=snippet["channelId"],
                channel_title=snippet["channelTitle"],
                thumbnail_url=snippet["thumbnails"]["high"]["url"],
                published_at=published_at,
                view_count=view_count,
                like_count=like_count,
                comment_count=comment_count,
                duration=content_details["duration"],
                category_id=snippet["categoryId"],
                tags=snippet.get("tags", []),
                description=snippet["description"],
                views_per_hour=views_per_hour,
                engagement_rate=engagement_rate
            )
            
        except Exception as e:
            logger.error(f"[BRAIN] Error parsing video data: {e}")
            return None
    
    def _deduplicate_videos(self, videos: List[VideoData]) -> List[VideoData]:
        """
        Remove duplicate videos based on video_id
        """
        seen_ids = set()
        unique_videos = []
        
        for video in videos:
            if video.video_id not in seen_ids:
                seen_ids.add(video.video_id)
                unique_videos.append(video)
        
        return unique_videos
    
    async def store_collected_data(self, all_data: Dict[str, List[VideoData]]):
        """
        Store collected data in Supabase for analysis
        """
        logger.info("[BRAIN] Storing collected data in database...")
        
        for niche, videos in all_data.items():
            if not videos:
                continue
                
            try:
                # Prepare data for database
                video_records = []
                for video in videos:
                    record = {
                        "video_id": video.video_id,
                        "title": video.title,
                        "channel_id": video.channel_id,
                        "channel_title": video.channel_title,
                        "thumbnail_url": video.thumbnail_url,
                        "published_at": video.published_at.isoformat(),
                        "view_count": video.view_count,
                        "like_count": video.like_count,
                        "comment_count": video.comment_count,
                        "duration": video.duration,
                        "category_id": video.category_id,
                        "tags": video.tags,
                        "description": video.description,
                        "views_per_hour": video.views_per_hour,
                        "engagement_rate": video.engagement_rate,
                        "niche": niche,
                        "collected_at": datetime.now().isoformat()
                    }
                    video_records.append(record)
                
                # Upsert to database
                result = self.supabase.table("youtube_videos").upsert(
                    video_records,
                    on_conflict="video_id"
                ).execute()
                
                logger.info(f"[BRAIN] Stored {len(video_records)} videos for {niche}")
                
            except Exception as e:
                logger.error(f"[BRAIN] Error storing {niche} data: {e}")

# Example usage
async def collect_youtube_data():
    """
    Main function to collect YouTube data
    """
    from supabase import create_client
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    
    if not all([supabase_url, supabase_key, youtube_key]):
        logger.error("[BRAIN] Missing required environment variables")
        return
    
    # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
    supabase = create_client(supabase_url, supabase_key)
    collector = YouTubeDataCollector(youtube_key, supabase)
    
    # Collect data
    all_data = await collector.collect_trending_data(max_videos_per_niche=200)
    
    # Store data
    await collector.store_collected_data(all_data)
    
    logger.info("[BRAIN] Data collection completed!")

if __name__ == "__main__":
    asyncio.run(collect_youtube_data())
