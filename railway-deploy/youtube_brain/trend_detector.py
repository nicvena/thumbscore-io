"""
YouTube Intelligence Brain - Trend Detector
Detects rising visual trends and emerging patterns in YouTube thumbnails
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class VisualTrend:
    trend_id: str
    niche: str
    trend_type: str  # "color", "composition", "text", "style"
    trend_strength: float
    growth_rate: float
    trend_description: str
    examples: List[str]
    predicted_lifespan: int  # days
    confidence: float

@dataclass
class TrendAlert:
    alert_type: str  # "emerging", "peaking", "declining"
    trend: VisualTrend
    urgency: str  # "low", "medium", "high"
    recommendation: str

class TrendDetector:
    """
    Detects and analyzes visual trends in YouTube thumbnails
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.trends = {}
        self.trend_history = defaultdict(list)
    
    async def detect_trends(self, niche: Optional[str] = None, days_back: int = 30) -> Dict[str, List[VisualTrend]]:
        """
        Detect visual trends across niches
        """
        logger.info(f"[TREND_DETECTOR] Detecting trends for niche: {niche or 'all'}, days_back: {days_back}")
        
        if niche:
            niches = [niche]
        else:
            niches = ["tech", "gaming", "education", "entertainment", "people", "music", "sports", "news", "comedy", "howto"]
        
        all_trends = {}
        
        for niche_name in niches:
            try:
                logger.info(f"[TREND_DETECTOR] Analyzing trends for {niche_name}")
                niche_trends = await self._detect_niche_trends(niche_name, days_back)
                all_trends[niche_name] = niche_trends
                
            except Exception as e:
                logger.error(f"[TREND_DETECTOR] Error detecting trends for {niche_name}: {e}")
                continue
        
        # Store trends
        await self._store_trends(all_trends)
        
        logger.info(f"[TREND_DETECTOR] Detected trends for {len(all_trends)} niches")
        return all_trends
    
    async def _detect_niche_trends(self, niche: str, days_back: int) -> List[VisualTrend]:
        """
        Detect trends for a specific niche
        """
        # Get recent videos with time series data
        videos = await self._get_recent_videos(niche, days_back)
        
        if len(videos) < 20:  # Need minimum data
            logger.warning(f"[TREND_DETECTOR] Insufficient data for {niche}: {len(videos)} videos")
            return []
        
        # Detect different types of trends
        trends = []
        
        # Color trends
        color_trends = await self._detect_color_trends(videos, niche)
        trends.extend(color_trends)
        
        # Text trends
        text_trends = await self._detect_text_trends(videos, niche)
        trends.extend(text_trends)
        
        # Composition trends
        composition_trends = await self._detect_composition_trends(videos, niche)
        trends.extend(composition_trends)
        
        # Style trends
        style_trends = await self._detect_style_trends(videos, niche)
        trends.extend(style_trends)
        
        # Sort by trend strength
        trends.sort(key=lambda t: t.trend_strength, reverse=True)
        
        logger.info(f"[TREND_DETECTOR] Found {len(trends)} trends for {niche}")
        return trends
    
    async def _get_recent_videos(self, niche: str, days_back: int) -> List[Dict[str, Any]]:
        """
        Get recent videos with performance data
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = self.supabase.table("youtube_videos").select("*").eq("niche", niche)
            query = query.gte("published_at", cutoff_date.isoformat())
            query = query.order("published_at", desc=True).limit(1000)
            
            result = query.execute()
            videos = result.data
            
            # Sort by publication date for trend analysis
            videos.sort(key=lambda v: v["published_at"])
            
            logger.info(f"[TREND_DETECTOR] Retrieved {len(videos)} recent videos for {niche}")
            return videos
            
        except Exception as e:
            logger.error(f"[TREND_DETECTOR] Error fetching recent videos: {e}")
            return []
    
    async def _detect_color_trends(self, videos: List[Dict[str, Any]], niche: str) -> List[VisualTrend]:
        """
        Detect color trends in thumbnails
        """
        trends = []
        
        # Analyze color patterns over time
        color_patterns = self._analyze_color_patterns(videos)
        
        for color_combo, pattern_data in color_patterns.items():
            if len(pattern_data["timeline"]) < 7:  # Need minimum data points
                continue
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(pattern_data["timeline"])
            growth_rate = self._calculate_growth_rate(pattern_data["timeline"])
            
            if trend_strength > 0.3:  # Significant trend
                trend = VisualTrend(
                    trend_id=f"{niche}_color_{color_combo}",
                    niche=niche,
                    trend_type="color",
                    trend_strength=trend_strength,
                    growth_rate=growth_rate,
                    trend_description=f"Rising {color_combo} color scheme",
                    examples=pattern_data["examples"][:3],
                    predicted_lifespan=self._predict_trend_lifespan(pattern_data["timeline"]),
                    confidence=min(trend_strength * 2, 1.0)
                )
                trends.append(trend)
        
        return trends
    
    def _analyze_color_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze color patterns in videos over time
        """
        color_patterns = defaultdict(lambda: {"timeline": [], "examples": []})
        
        for video in videos:
            # Extract color information from title/tags (simplified)
            color_keywords = self._extract_color_keywords(video["title"], video["tags"])
            
            for color in color_keywords:
                color_patterns[color]["timeline"].append({
                    "date": video["published_at"],
                    "success": video["views_per_hour"],
                    "engagement": video["engagement_rate"]
                })
                color_patterns[color]["examples"].append(video["thumbnail_url"])
        
        return dict(color_patterns)
    
    def _extract_color_keywords(self, title: str, tags: List[str]) -> List[str]:
        """
        Extract color-related keywords from title and tags
        """
        color_keywords = []
        
        # Common color words
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "dark", "bright", "vibrant"]
        
        text = (title + " " + " ".join(tags)).lower()
        
        for color in colors:
            if color in text:
                color_keywords.append(color)
        
        return color_keywords
    
    async def _detect_text_trends(self, videos: List[Dict[str, Any]], niche: str) -> List[VisualTrend]:
        """
        Detect text and typography trends
        """
        trends = []
        
        # Analyze text patterns over time
        text_patterns = self._analyze_text_patterns(videos)
        
        for pattern_type, pattern_data in text_patterns.items():
            if len(pattern_data["timeline"]) < 7:
                continue
            
            trend_strength = self._calculate_trend_strength(pattern_data["timeline"])
            growth_rate = self._calculate_growth_rate(pattern_data["timeline"])
            
            if trend_strength > 0.3:
                trend = VisualTrend(
                    trend_id=f"{niche}_text_{pattern_type}",
                    niche=niche,
                    trend_type="text",
                    trend_strength=trend_strength,
                    growth_rate=growth_rate,
                    trend_description=f"Rising {pattern_type} text style",
                    examples=pattern_data["examples"][:3],
                    predicted_lifespan=self._predict_trend_lifespan(pattern_data["timeline"]),
                    confidence=min(trend_strength * 2, 1.0)
                )
                trends.append(trend)
        
        return trends
    
    def _analyze_text_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze text patterns in videos over time
        """
        text_patterns = defaultdict(lambda: {"timeline": [], "examples": []})
        
        for video in videos:
            title = video["title"]
            
            # Analyze different text characteristics
            patterns = {
                "caps_heavy": sum(1 for c in title if c.isupper()) / len(title) > 0.5,
                "question_titles": "?" in title,
                "exclamation_titles": "!" in title,
                "number_titles": any(c.isdigit() for c in title),
                "short_titles": len(title) < 50,
                "long_titles": len(title) > 80,
                "emoji_titles": any(ord(c) > 127 for c in title)
            }
            
            for pattern_type, is_present in patterns.items():
                if is_present:
                    text_patterns[pattern_type]["timeline"].append({
                        "date": video["published_at"],
                        "success": video["views_per_hour"],
                        "engagement": video["engagement_rate"]
                    })
                    text_patterns[pattern_type]["examples"].append(video["thumbnail_url"])
        
        return dict(text_patterns)
    
    async def _detect_composition_trends(self, videos: List[Dict[str, Any]], niche: str) -> List[VisualTrend]:
        """
        Detect composition and layout trends
        """
        trends = []
        
        # Analyze composition patterns
        composition_patterns = self._analyze_composition_patterns(videos)
        
        for comp_type, pattern_data in composition_patterns.items():
            if len(pattern_data["timeline"]) < 7:
                continue
            
            trend_strength = self._calculate_trend_strength(pattern_data["timeline"])
            growth_rate = self._calculate_growth_rate(pattern_data["timeline"])
            
            if trend_strength > 0.3:
                trend = VisualTrend(
                    trend_id=f"{niche}_composition_{comp_type}",
                    niche=niche,
                    trend_type="composition",
                    trend_strength=trend_strength,
                    growth_rate=growth_rate,
                    trend_description=f"Rising {comp_type} composition style",
                    examples=pattern_data["examples"][:3],
                    predicted_lifespan=self._predict_trend_lifespan(pattern_data["timeline"]),
                    confidence=min(trend_strength * 2, 1.0)
                )
                trends.append(trend)
        
        return trends
    
    def _analyze_composition_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze composition patterns in videos over time
        """
        composition_patterns = defaultdict(lambda: {"timeline": [], "examples": []})
        
        for video in videos:
            title = video["title"]
            
            # Analyze composition-related keywords
            composition_keywords = {
                "face_closeup": ["face", "close", "up", "portrait", "selfie"],
                "split_screen": ["vs", "versus", "comparison", "split", "side"],
                "before_after": ["before", "after", "transformation", "change"],
                "reaction": ["reaction", "react", "shocked", "surprised"],
                "tutorial": ["how", "tutorial", "guide", "step", "learn"],
                "review": ["review", "test", "unboxing", "first look"]
            }
            
            for comp_type, keywords in composition_keywords.items():
                if any(keyword in title.lower() for keyword in keywords):
                    composition_patterns[comp_type]["timeline"].append({
                        "date": video["published_at"],
                        "success": video["views_per_hour"],
                        "engagement": video["engagement_rate"]
                    })
                    composition_patterns[comp_type]["examples"].append(video["thumbnail_url"])
        
        return dict(composition_patterns)
    
    async def _detect_style_trends(self, videos: List[Dict[str, Any]], niche: str) -> List[VisualTrend]:
        """
        Detect overall style trends
        """
        trends = []
        
        # Analyze style patterns
        style_patterns = self._analyze_style_patterns(videos)
        
        for style_type, pattern_data in style_patterns.items():
            if len(pattern_data["timeline"]) < 7:
                continue
            
            trend_strength = self._calculate_trend_strength(pattern_data["timeline"])
            growth_rate = self._calculate_growth_rate(pattern_data["timeline"])
            
            if trend_strength > 0.3:
                trend = VisualTrend(
                    trend_id=f"{niche}_style_{style_type}",
                    niche=niche,
                    trend_type="style",
                    trend_strength=trend_strength,
                    growth_rate=growth_rate,
                    trend_description=f"Rising {style_type} style trend",
                    examples=pattern_data["examples"][:3],
                    predicted_lifespan=self._predict_trend_lifespan(pattern_data["timeline"]),
                    confidence=min(trend_strength * 2, 1.0)
                )
                trends.append(trend)
        
        return trends
    
    def _analyze_style_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze style patterns in videos over time
        """
        style_patterns = defaultdict(lambda: {"timeline": [], "examples": []})
        
        for video in videos:
            # Analyze style indicators
            duration_seconds = self._parse_duration(video["duration"])
            
            styles = {
                "short_form": duration_seconds < 300,
                "long_form": duration_seconds > 1800,
                "high_engagement": video["engagement_rate"] > 0.05,
                "viral_potential": video["views_per_hour"] > 1000,
                "trending_topic": any(keyword in video["title"].lower() for keyword in ["2024", "2025", "new", "latest", "breaking"])
            }
            
            for style_type, is_present in styles.items():
                if is_present:
                    style_patterns[style_type]["timeline"].append({
                        "date": video["published_at"],
                        "success": video["views_per_hour"],
                        "engagement": video["engagement_rate"]
                    })
                    style_patterns[style_type]["examples"].append(video["thumbnail_url"])
        
        return dict(style_patterns)
    
    def _calculate_trend_strength(self, timeline: List[Dict[str, Any]]) -> float:
        """
        Calculate trend strength using linear regression
        """
        if len(timeline) < 3:
            return 0.0
        
        # Extract success metrics over time
        successes = [point["success"] for point in timeline]
        
        # Calculate trend using simple linear regression
        x = np.arange(len(successes))
        y = np.array(successes)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to 0-1 range
        max_success = max(successes)
        trend_strength = abs(slope) / max_success if max_success > 0 else 0
        
        return min(trend_strength, 1.0)
    
    def _calculate_growth_rate(self, timeline: List[Dict[str, Any]]) -> float:
        """
        Calculate growth rate of trend
        """
        if len(timeline) < 2:
            return 0.0
        
        successes = [point["success"] for point in timeline]
        
        # Calculate percentage growth
        initial = successes[0]
        final = successes[-1]
        
        if initial == 0:
            return 0.0
        
        growth_rate = (final - initial) / initial
        return growth_rate
    
    def _predict_trend_lifespan(self, timeline: List[Dict[str, Any]]) -> int:
        """
        Predict how long a trend will last based on historical data
        """
        if len(timeline) < 5:
            return 30  # Default prediction
        
        # Analyze trend velocity
        trend_strength = self._calculate_trend_strength(timeline)
        growth_rate = self._calculate_growth_rate(timeline)
        
        # Predict lifespan based on trend characteristics
        if trend_strength > 0.7 and growth_rate > 0.5:
            return 60  # Strong growing trend
        elif trend_strength > 0.5:
            return 45  # Moderate trend
        elif trend_strength > 0.3:
            return 30  # Weak trend
        else:
            return 15  # Very weak trend
    
    def _parse_duration(self, duration: str) -> float:
        """
        Parse ISO 8601 duration to seconds
        """
        if not duration.startswith("PT"):
            return 0
        
        duration = duration[2:]  # Remove PT
        
        seconds = 0
        if "H" in duration:
            hours = int(duration.split("H")[0])
            seconds += hours * 3600
            duration = duration.split("H")[1]
        
        if "M" in duration:
            minutes = int(duration.split("M")[0])
            seconds += minutes * 60
            duration = duration.split("M")[1]
        
        if "S" in duration:
            seconds += int(duration.split("S")[0])
        
        return seconds
    
    async def _store_trends(self, all_trends: Dict[str, List[VisualTrend]]):
        """
        Store detected trends in database
        """
        try:
            records = []
            for niche, trends in all_trends.items():
                for trend in trends:
                    record = {
                        "trend_id": trend.trend_id,
                        "niche": trend.niche,
                        "trend_type": trend.trend_type,
                        "trend_strength": trend.trend_strength,
                        "growth_rate": trend.growth_rate,
                        "trend_description": trend.trend_description,
                        "examples": trend.examples,
                        "predicted_lifespan": trend.predicted_lifespan,
                        "confidence": trend.confidence,
                        "created_at": "now()"
                    }
                    records.append(record)
            
            if records:
                self.supabase.table("visual_trends").upsert(records, on_conflict="trend_id").execute()
                logger.info(f"[TREND_DETECTOR] Stored {len(records)} trends")
            
        except Exception as e:
            logger.error(f"[TREND_DETECTOR] Error storing trends: {e}")
    
    async def get_trend_alerts(self, niche: Optional[str] = None) -> List[TrendAlert]:
        """
        Get trend alerts for creators
        """
        try:
            query = self.supabase.table("visual_trends").select("*")
            if niche:
                query = query.eq("niche", niche)
            
            query = query.order("trend_strength", desc=True).limit(20)
            result = query.execute()
            trends_data = result.data
            
            alerts = []
            for trend_data in trends_data:
                trend = VisualTrend(
                    trend_id=trend_data["trend_id"],
                    niche=trend_data["niche"],
                    trend_type=trend_data["trend_type"],
                    trend_strength=trend_data["trend_strength"],
                    growth_rate=trend_data["growth_rate"],
                    trend_description=trend_data["trend_description"],
                    examples=trend_data["examples"],
                    predicted_lifespan=trend_data["predicted_lifespan"],
                    confidence=trend_data["confidence"]
                )
                
                # Determine alert type and urgency
                if trend.trend_strength > 0.7 and trend.growth_rate > 0.5:
                    alert_type = "emerging"
                    urgency = "high"
                    recommendation = f"🚀 Jump on this {trend.trend_type} trend now! It's growing fast and has strong momentum."
                elif trend.trend_strength > 0.5:
                    alert_type = "emerging"
                    urgency = "medium"
                    recommendation = f"📈 Consider adopting this {trend.trend_type} trend. It's showing good growth."
                elif trend.trend_strength > 0.3:
                    alert_type = "emerging"
                    urgency = "low"
                    recommendation = f"👀 Watch this {trend.trend_type} trend. It's emerging but not yet strong."
                else:
                    continue
                
                alert = TrendAlert(
                    alert_type=alert_type,
                    trend=trend,
                    urgency=urgency,
                    recommendation=recommendation
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"[TREND_DETECTOR] Error getting trend alerts: {e}")
            return []

# Example usage
async def detect_trends():
    """
    Main function to detect trends
    """
    from supabase import create_client
    import os
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not all([supabase_url, supabase_key]):
        logger.error("[TREND_DETECTOR] Missing Supabase credentials")
        return
    
    # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
    supabase = create_client(supabase_url, supabase_key)
    detector = TrendDetector(supabase)
    
    # Detect trends for all niches
    all_trends = await detector.detect_trends(days_back=30)
    
    logger.info(f"[TREND_DETECTOR] Detected trends for {len(all_trends)} niches")
    for niche, trends in all_trends.items():
        logger.info(f"[TREND_DETECTOR] {niche}: {len(trends)} trends")
    
    # Get trend alerts
    alerts = await detector.get_trend_alerts()
    logger.info(f"[TREND_DETECTOR] Generated {len(alerts)} trend alerts")

if __name__ == "__main__":
    import asyncio
    asyncio.run(detect_trends())
