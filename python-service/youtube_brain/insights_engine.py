"""
YouTube Intelligence Brain - Insights Engine
Aggregates per-creator insights and provides personalized recommendations
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
class CreatorInsight:
    channel_id: str
    channel_name: str
    niche: str
    total_videos: int
    avg_performance: Dict[str, float]
    best_performing_patterns: List[Dict[str, Any]]
    improvement_opportunities: List[str]
    competitor_analysis: Dict[str, Any]
    personalized_recommendations: List[str]
    performance_trends: Dict[str, Any]
    last_updated: datetime

@dataclass
class PerformanceInsight:
    metric_name: str
    current_value: float
    benchmark_value: float
    percentile: float
    trend_direction: str  # "up", "down", "stable"
    recommendation: str

class InsightsEngine:
    """
    Generates personalized insights for YouTube creators
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.creator_insights = {}
        self.benchmarks = {}
    
    async def generate_creator_insights(self, channel_id: Optional[str] = None) -> Dict[str, CreatorInsight]:
        """
        Generate insights for specific creator or all creators
        """
        logger.info(f"[INSIGHTS_ENGINE] Generating insights for channel: {channel_id or 'all'}")
        
        if channel_id:
            channels = [channel_id]
        else:
            channels = await self._get_active_channels()
        
        insights = {}
        
        for ch_id in channels:
            try:
                logger.info(f"[INSIGHTS_ENGINE] Analyzing channel {ch_id}")
                insight = await self._analyze_creator_performance(ch_id)
                insights[ch_id] = insight
                
            except Exception as e:
                logger.error(f"[INSIGHTS_ENGINE] Error analyzing channel {ch_id}: {e}")
                continue
        
        # Store insights
        await self._store_creator_insights(insights)
        
        logger.info(f"[INSIGHTS_ENGINE] Generated insights for {len(insights)} creators")
        return insights
    
    async def _get_active_channels(self) -> List[str]:
        """
        Get list of active channels with recent videos
        """
        try:
            # Get channels with videos in the last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            
            query = self.supabase.table("youtube_videos").select("channel_id").distinct()
            query = query.gte("published_at", cutoff_date.isoformat())
            
            result = query.execute()
            channels = [row["channel_id"] for row in result.data]
            
            logger.info(f"[INSIGHTS_ENGINE] Found {len(channels)} active channels")
            return channels
            
        except Exception as e:
            logger.error(f"[INSIGHTS_ENGINE] Error fetching active channels: {e}")
            return []
    
    async def _analyze_creator_performance(self, channel_id: str) -> CreatorInsight:
        """
        Analyze performance for a specific creator
        """
        # Get creator's videos
        videos = await self._get_creator_videos(channel_id)
        
        if not videos:
            logger.warning(f"[INSIGHTS_ENGINE] No videos found for channel {channel_id}")
            return CreatorInsight(
                channel_id=channel_id,
                channel_name="Unknown",
                niche="unknown",
                total_videos=0,
                avg_performance={},
                best_performing_patterns=[],
                improvement_opportunities=[],
                competitor_analysis={},
                personalized_recommendations=[],
                performance_trends={},
                last_updated=datetime.now()
            )
        
        # Analyze performance metrics
        avg_performance = self._calculate_avg_performance(videos)
        
        # Find best performing patterns
        best_patterns = await self._find_best_performing_patterns(videos)
        
        # Identify improvement opportunities
        improvement_opportunities = await self._identify_improvement_opportunities(videos, avg_performance)
        
        # Competitor analysis
        competitor_analysis = await self._analyze_competitors(channel_id, videos[0]["niche"])
        
        # Generate personalized recommendations
        recommendations = await self._generate_personalized_recommendations(
            videos, avg_performance, best_patterns, improvement_opportunities
        )
        
        # Analyze performance trends
        performance_trends = self._analyze_performance_trends(videos)
        
        return CreatorInsight(
            channel_id=channel_id,
            channel_name=videos[0]["channel_title"],
            niche=videos[0]["niche"],
            total_videos=len(videos),
            avg_performance=avg_performance,
            best_performing_patterns=best_patterns,
            improvement_opportunities=improvement_opportunities,
            competitor_analysis=competitor_analysis,
            personalized_recommendations=recommendations,
            performance_trends=performance_trends,
            last_updated=datetime.now()
        )
    
    async def _get_creator_videos(self, channel_id: str) -> List[Dict[str, Any]]:
        """
        Get all videos for a creator
        """
        try:
            query = self.supabase.table("youtube_videos").select("*").eq("channel_id", channel_id)
            query = query.order("published_at", desc=True).limit(100)  # Last 100 videos
            
            result = query.execute()
            videos = result.data
            
            logger.info(f"[INSIGHTS_ENGINE] Retrieved {len(videos)} videos for channel {channel_id}")
            return videos
            
        except Exception as e:
            logger.error(f"[INSIGHTS_ENGINE] Error fetching videos for channel {channel_id}: {e}")
            return []
    
    def _calculate_avg_performance(self, videos: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average performance metrics
        """
        if not videos:
            return {}
        
        metrics = {
            "avg_views_per_hour": np.mean([v["views_per_hour"] for v in videos]),
            "avg_engagement_rate": np.mean([v["engagement_rate"] for v in videos]),
            "avg_view_count": np.mean([v["view_count"] for v in videos]),
            "avg_like_count": np.mean([v["like_count"] for v in videos]),
            "avg_comment_count": np.mean([v["comment_count"] for v in videos]),
            "avg_title_length": np.mean([len(v["title"]) for v in videos]),
            "avg_tag_count": np.mean([len(v["tags"]) for v in videos]),
            "caps_percentage": np.mean([sum(1 for c in v["title"] if c.isupper()) / len(v["title"]) for v in videos if v["title"]]),
            "question_rate": np.mean([1 if "?" in v["title"] else 0 for v in videos]),
            "exclamation_rate": np.mean([1 if "!" in v["title"] else 0 for v in videos])
        }
        
        return metrics
    
    async def _find_best_performing_patterns(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find patterns in best performing videos
        """
        if len(videos) < 5:
            return []
        
        # Sort by performance
        top_performers = sorted(videos, key=lambda v: v["views_per_hour"], reverse=True)[:5]
        
        patterns = []
        
        # Title patterns
        title_patterns = self._analyze_title_patterns(top_performers)
        if title_patterns:
            patterns.append({
                "type": "title_patterns",
                "description": "Common patterns in your best performing titles",
                "patterns": title_patterns
            })
        
        # Content patterns
        content_patterns = self._analyze_content_patterns(top_performers)
        if content_patterns:
            patterns.append({
                "type": "content_patterns",
                "description": "Content characteristics of your top videos",
                "patterns": content_patterns
            })
        
        # Timing patterns
        timing_patterns = self._analyze_timing_patterns(top_performers)
        if timing_patterns:
            patterns.append({
                "type": "timing_patterns",
                "description": "Optimal timing for your content",
                "patterns": timing_patterns
            })
        
        return patterns
    
    def _analyze_title_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in video titles
        """
        titles = [v["title"] for v in videos]
        
        patterns = {
            "common_words": self._find_common_words(titles),
            "avg_length": np.mean([len(title) for title in titles]),
            "caps_usage": np.mean([sum(1 for c in title if c.isupper()) / len(title) for title in titles if title]),
            "question_usage": sum(1 for title in titles if "?" in title) / len(titles),
            "exclamation_usage": sum(1 for title in titles if "!" in title) / len(titles),
            "number_usage": sum(1 for title in titles if any(c.isdigit() for c in title)) / len(titles)
        }
        
        return patterns
    
    def _find_common_words(self, titles: List[str]) -> List[Tuple[str, int]]:
        """
        Find most common words in titles
        """
        word_count = Counter()
        for title in titles:
            words = title.lower().split()
            word_count.update(words)
        
        # Remove stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "this", "that", "these", "those"}
        
        filtered_words = [(word, count) for word, count in word_count.most_common(20) if word not in stop_words and len(word) > 2]
        
        return filtered_words[:10]
    
    def _analyze_content_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze content patterns in videos
        """
        patterns = {
            "avg_duration": np.mean([self._parse_duration(v["duration"]) for v in videos]),
            "duration_categories": Counter([self._categorize_duration(v["duration"]) for v in videos]),
            "avg_tag_count": np.mean([len(v["tags"]) for v in videos]),
            "common_tags": self._find_common_tags([v["tags"] for v in videos]),
            "engagement_patterns": {
                "avg_like_rate": np.mean([v["like_count"] / max(v["view_count"], 1) for v in videos]),
                "avg_comment_rate": np.mean([v["comment_count"] / max(v["view_count"], 1) for v in videos])
            }
        }
        
        return patterns
    
    def _categorize_duration(self, duration: str) -> str:
        """
        Categorize video duration
        """
        seconds = self._parse_duration(duration)
        
        if seconds < 300:
            return "short"
        elif seconds < 900:
            return "medium"
        elif seconds < 1800:
            return "long"
        else:
            return "very_long"
    
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
    
    def _find_common_tags(self, tags_lists: List[List[str]]) -> List[Tuple[str, int]]:
        """
        Find most common tags across videos
        """
        tag_count = Counter()
        for tags in tags_lists:
            tag_count.update(tags)
        
        return tag_count.most_common(10)
    
    def _analyze_timing_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze timing patterns in video performance
        """
        patterns = {
            "best_days": Counter([datetime.fromisoformat(v["published_at"].replace("Z", "+00:00")).strftime("%A") for v in videos]),
            "best_hours": Counter([datetime.fromisoformat(v["published_at"].replace("Z", "+00:00")).hour for v in videos]),
            "seasonal_patterns": self._analyze_seasonal_patterns(videos)
        }
        
        return patterns
    
    def _analyze_seasonal_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in video performance
        """
        monthly_performance = defaultdict(list)
        
        for video in videos:
            published_date = datetime.fromisoformat(video["published_at"].replace("Z", "+00:00"))
            month = published_date.strftime("%B")
            monthly_performance[month].append(video["views_per_hour"])
        
        seasonal_patterns = {}
        for month, performances in monthly_performance.items():
            seasonal_patterns[month] = {
                "avg_performance": np.mean(performances),
                "video_count": len(performances)
            }
        
        return seasonal_patterns
    
    async def _identify_improvement_opportunities(self, videos: List[Dict[str, Any]], avg_performance: Dict[str, float]) -> List[str]:
        """
        Identify improvement opportunities for the creator
        """
        opportunities = []
        
        # Compare with niche benchmarks
        niche = videos[0]["niche"] if videos else "unknown"
        benchmarks = await self._get_niche_benchmarks(niche)
        
        if benchmarks:
            # Title length optimization
            if avg_performance.get("avg_title_length", 0) < benchmarks.get("optimal_title_length", 50):
                opportunities.append("Consider longer, more descriptive titles (current: {:.0f} chars, optimal: {:.0f} chars)".format(
                    avg_performance.get("avg_title_length", 0), benchmarks.get("optimal_title_length", 50)
                ))
            
            # Engagement rate improvement
            if avg_performance.get("avg_engagement_rate", 0) < benchmarks.get("avg_engagement_rate", 0.03):
                opportunities.append("Improve engagement rate (current: {:.1%}, benchmark: {:.1%})".format(
                    avg_performance.get("avg_engagement_rate", 0), benchmarks.get("avg_engagement_rate", 0.03)
                ))
            
            # Caps usage optimization
            if avg_performance.get("caps_percentage", 0) > benchmarks.get("optimal_caps_percentage", 0.3):
                opportunities.append("Reduce caps usage (current: {:.1%}, optimal: {:.1%})".format(
                    avg_performance.get("caps_percentage", 0), benchmarks.get("optimal_caps_percentage", 0.3)
                ))
        
        # Content-specific opportunities
        if avg_performance.get("question_rate", 0) < 0.2:
            opportunities.append("Try more question-based titles to increase curiosity")
        
        if avg_performance.get("exclamation_rate", 0) < 0.1:
            opportunities.append("Add more excitement with exclamation marks")
        
        if avg_performance.get("avg_tag_count", 0) < 10:
            opportunities.append("Add more tags to improve discoverability")
        
        return opportunities
    
    async def _get_niche_benchmarks(self, niche: str) -> Dict[str, float]:
        """
        Get performance benchmarks for a niche
        """
        try:
            query = self.supabase.table("youtube_videos").select("*").eq("niche", niche)
            result = query.execute()
            videos = result.data
            
            if not videos:
                return {}
            
            benchmarks = {
                "avg_views_per_hour": np.mean([v["views_per_hour"] for v in videos]),
                "avg_engagement_rate": np.mean([v["engagement_rate"] for v in videos]),
                "avg_title_length": np.mean([len(v["title"]) for v in videos]),
                "avg_caps_percentage": np.mean([sum(1 for c in v["title"] if c.isupper()) / len(v["title"]) for v in videos if v["title"]]),
                "avg_tag_count": np.mean([len(v["tags"]) for v in videos]),
                "optimal_title_length": 60,  # Based on research
                "optimal_caps_percentage": 0.3,  # Based on research
                "optimal_tag_count": 15  # Based on research
            }
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"[INSIGHTS_ENGINE] Error getting benchmarks for {niche}: {e}")
            return {}
    
    async def _analyze_competitors(self, channel_id: str, niche: str) -> Dict[str, Any]:
        """
        Analyze competitors in the same niche
        """
        try:
            # Get top performers in the same niche
            query = self.supabase.table("youtube_videos").select("*").eq("niche", niche)
            query = query.order("views_per_hour", desc=True).limit(100)
            
            result = query.execute()
            videos = result.data
            
            # Filter out the creator's own videos
            competitor_videos = [v for v in videos if v["channel_id"] != channel_id]
            
            if not competitor_videos:
                return {}
            
            # Analyze competitor performance
            competitor_analysis = {
                "top_competitors": self._get_top_competitors(competitor_videos),
                "competitor_benchmarks": self._calculate_competitor_benchmarks(competitor_videos),
                "competitive_gaps": self._identify_competitive_gaps(competitor_videos, channel_id)
            }
            
            return competitor_analysis
            
        except Exception as e:
            logger.error(f"[INSIGHTS_ENGINE] Error analyzing competitors: {e}")
            return {}
    
    def _get_top_competitors(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get top performing competitors
        """
        # Group by channel
        channel_performance = defaultdict(list)
        for video in videos:
            channel_performance[video["channel_id"]].append(video["views_per_hour"])
        
        # Calculate average performance per channel
        competitor_scores = []
        for channel_id, performances in channel_performance.items():
            avg_performance = np.mean(performances)
            competitor_scores.append({
                "channel_id": channel_id,
                "channel_name": videos[0]["channel_title"] if videos else "Unknown",
                "avg_views_per_hour": avg_performance,
                "video_count": len(performances)
            })
        
        # Sort by performance
        competitor_scores.sort(key=lambda x: x["avg_views_per_hour"], reverse=True)
        
        return competitor_scores[:5]
    
    def _calculate_competitor_benchmarks(self, videos: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate competitor benchmarks
        """
        return {
            "avg_views_per_hour": np.mean([v["views_per_hour"] for v in videos]),
            "avg_engagement_rate": np.mean([v["engagement_rate"] for v in videos]),
            "avg_title_length": np.mean([len(v["title"]) for v in videos]),
            "avg_tag_count": np.mean([len(v["tags"]) for v in videos])
        }
    
    def _identify_competitive_gaps(self, videos: List[Dict[str, Any]], channel_id: str) -> List[str]:
        """
        Identify competitive gaps
        """
        gaps = []
        
        # Analyze common patterns in top competitor videos
        top_competitors = sorted(videos, key=lambda v: v["views_per_hour"], reverse=True)[:20]
        
        # Title patterns
        competitor_titles = [v["title"] for v in top_competitors]
        common_words = self._find_common_words(competitor_titles)
        
        if common_words:
            gaps.append(f"Top competitors use these words: {', '.join([word for word, count in common_words[:5]])}")
        
        # Content patterns
        competitor_durations = [self._parse_duration(v["duration"]) for v in top_competitors]
        avg_duration = np.mean(competitor_durations)
        
        if avg_duration < 600:  # Less than 10 minutes
            gaps.append("Top competitors favor shorter content (avg: {:.0f} minutes)".format(avg_duration / 60))
        
        return gaps
    
    async def _generate_personalized_recommendations(self, videos: List[Dict[str, Any]], avg_performance: Dict[str, float], best_patterns: List[Dict[str, Any]], improvement_opportunities: List[str]) -> List[str]:
        """
        Generate personalized recommendations for the creator
        """
        recommendations = []
        
        # Performance-based recommendations
        if avg_performance.get("avg_views_per_hour", 0) < 100:
            recommendations.append("🚀 Focus on creating more engaging content to increase view velocity")
        
        if avg_performance.get("avg_engagement_rate", 0) < 0.03:
            recommendations.append("💬 Improve audience engagement with more interactive content")
        
        # Pattern-based recommendations
        if best_patterns:
            for pattern in best_patterns:
                if pattern["type"] == "title_patterns":
                    common_words = pattern["patterns"].get("common_words", [])
                    if common_words:
                        recommendations.append(f"✨ Your best titles use these words: {', '.join([word for word, count in common_words[:3]])}")
        
        # Improvement opportunities
        recommendations.extend(improvement_opportunities[:3])  # Top 3 opportunities
        
        # Trend-based recommendations
        recommendations.append("📈 Stay updated with trending topics in your niche")
        recommendations.append("🎯 A/B test different thumbnail styles to find what works best")
        
        return recommendations
    
    def _analyze_performance_trends(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance trends over time
        """
        if len(videos) < 10:
            return {}
        
        # Sort by publication date
        videos.sort(key=lambda v: v["published_at"])
        
        # Calculate rolling averages
        window_size = min(5, len(videos) // 2)
        views_trend = []
        engagement_trend = []
        
        for i in range(window_size, len(videos)):
            window_videos = videos[i-window_size:i]
            views_trend.append(np.mean([v["views_per_hour"] for v in window_videos]))
            engagement_trend.append(np.mean([v["engagement_rate"] for v in window_videos]))
        
        # Calculate trend direction
        views_slope = np.polyfit(range(len(views_trend)), views_trend, 1)[0] if views_trend else 0
        engagement_slope = np.polyfit(range(len(engagement_trend)), engagement_trend, 1)[0] if engagement_trend else 0
        
        return {
            "views_trend": "up" if views_slope > 0 else "down" if views_slope < 0 else "stable",
            "engagement_trend": "up" if engagement_slope > 0 else "down" if engagement_slope < 0 else "stable",
            "views_slope": views_slope,
            "engagement_slope": engagement_slope,
            "recent_performance": {
                "views_per_hour": views_trend[-1] if views_trend else 0,
                "engagement_rate": engagement_trend[-1] if engagement_trend else 0
            }
        }
    
    async def _store_creator_insights(self, insights: Dict[str, CreatorInsight]):
        """
        Store creator insights in database
        """
        try:
            records = []
            for channel_id, insight in insights.items():
                record = {
                    "channel_id": insight.channel_id,
                    "channel_name": insight.channel_name,
                    "niche": insight.niche,
                    "total_videos": insight.total_videos,
                    "avg_performance": insight.avg_performance,
                    "best_performing_patterns": insight.best_performing_patterns,
                    "improvement_opportunities": insight.improvement_opportunities,
                    "competitor_analysis": insight.competitor_analysis,
                    "personalized_recommendations": insight.personalized_recommendations,
                    "performance_trends": insight.performance_trends,
                    "last_updated": insight.last_updated.isoformat()
                }
                records.append(record)
            
            if records:
                self.supabase.table("creator_insights").upsert(records, on_conflict="channel_id").execute()
                logger.info(f"[INSIGHTS_ENGINE] Stored insights for {len(records)} creators")
            
        except Exception as e:
            logger.error(f"[INSIGHTS_ENGINE] Error storing creator insights: {e}")
    
    async def get_creator_insights(self, channel_id: str) -> Optional[CreatorInsight]:
        """
        Get stored insights for a specific creator
        """
        try:
            query = self.supabase.table("creator_insights").select("*").eq("channel_id", channel_id)
            result = query.execute()
            
            if not result.data:
                return None
            
            data = result.data[0]
            
            return CreatorInsight(
                channel_id=data["channel_id"],
                channel_name=data["channel_name"],
                niche=data["niche"],
                total_videos=data["total_videos"],
                avg_performance=data["avg_performance"],
                best_performing_patterns=data["best_performing_patterns"],
                improvement_opportunities=data["improvement_opportunities"],
                competitor_analysis=data["competitor_analysis"],
                personalized_recommendations=data["personalized_recommendations"],
                performance_trends=data["performance_trends"],
                last_updated=datetime.fromisoformat(data["last_updated"])
            )
            
        except Exception as e:
            logger.error(f"[INSIGHTS_ENGINE] Error getting creator insights: {e}")
            return None

# Example usage
async def generate_insights():
    """
    Main function to generate creator insights
    """
    from supabase import create_client
    import os
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not all([supabase_url, supabase_key]):
        logger.error("[INSIGHTS_ENGINE] Missing Supabase credentials")
        return
    
    # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
    supabase = create_client(supabase_url, supabase_key)
    engine = InsightsEngine(supabase)
    
    # Generate insights for all active creators
    insights = await engine.generate_creator_insights()
    
    logger.info(f"[INSIGHTS_ENGINE] Generated insights for {len(insights)} creators")
    for channel_id, insight in insights.items():
        logger.info(f"[INSIGHTS_ENGINE] {insight.channel_name}: {len(insight.personalized_recommendations)} recommendations")

if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_insights())
