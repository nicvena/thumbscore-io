"""
YouTube Intelligence Brain - Pattern Miner
Clusters embeddings and finds recurring visual patterns in successful thumbnails
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import Counter
import json

logger = logging.getLogger(__name__)

@dataclass
class VisualPattern:
    pattern_id: str
    niche: str
    cluster_center: List[float]
    success_rate: float
    avg_views_per_hour: float
    common_features: Dict[str, Any]
    thumbnail_examples: List[str]
    pattern_description: str

@dataclass
class FeaturePattern:
    feature_name: str
    feature_type: str  # "color", "composition", "text", "face", etc.
    success_threshold: float
    impact_score: float
    examples: List[str]

class PatternMiner:
    """
    Mines visual patterns from successful YouTube thumbnails
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.patterns = {}
        self.feature_patterns = {}
    
    async def mine_patterns(self, niche: Optional[str] = None) -> Dict[str, List[VisualPattern]]:
        """
        Mine visual patterns from collected YouTube data
        """
        logger.info(f"[PATTERN_MINER] Starting pattern mining for niche: {niche or 'all'}")
        
        # Get high-performing videos
        videos = await self._get_high_performing_videos(niche)
        
        if not videos:
            logger.warning("[PATTERN_MINER] No videos found for pattern mining")
            return {}
        
        # Extract embeddings and features
        embeddings, features = await self._extract_embeddings_and_features(videos)
        
        if len(embeddings) < 10:
            logger.warning("[PATTERN_MINER] Not enough data for pattern mining")
            return {}
        
        # Cluster embeddings to find visual patterns
        visual_patterns = await self._cluster_embeddings(embeddings, videos, niche)
        
        # Mine feature patterns
        feature_patterns = await self._mine_feature_patterns(features, videos)
        
        # Store patterns
        await self._store_patterns(visual_patterns, feature_patterns, niche)
        
        logger.info(f"[PATTERN_MINER] Found {len(visual_patterns)} visual patterns and {len(feature_patterns)} feature patterns")
        
        return {
            "visual_patterns": visual_patterns,
            "feature_patterns": feature_patterns
        }
    
    async def _get_high_performing_videos(self, niche: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get high-performing videos for pattern mining
        """
        try:
            query = self.supabase.table("youtube_videos").select("*")
            
            if niche:
                query = query.eq("niche", niche)
            
            # Get videos with high views_per_hour (top 20%)
            query = query.order("views_per_hour", desc=True).limit(1000)
            
            result = query.execute()
            videos = result.data
            
            # Filter to top performers (above 75th percentile)
            if videos:
                views_per_hour = [v["views_per_hour"] for v in videos]
                threshold = np.percentile(views_per_hour, 75)
                videos = [v for v in videos if v["views_per_hour"] >= threshold]
            
            logger.info(f"[PATTERN_MINER] Found {len(videos)} high-performing videos")
            return videos
            
        except Exception as e:
            logger.error(f"[PATTERN_MINER] Error fetching videos: {e}")
            return []
    
    async def _extract_embeddings_and_features(self, videos: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Extract CLIP embeddings and visual features from videos
        """
        embeddings = []
        features = []
        
        for video in videos:
            try:
                # Get CLIP embedding from ref_thumbnails table
                embedding_result = self.supabase.table("ref_thumbnails").select("embedding").eq("thumb_url", video["thumbnail_url"]).execute()
                
                if embedding_result.data:
                    embedding = embedding_result.data[0]["embedding"]
                    embeddings.append(embedding)
                    
                    # Extract visual features
                    feature = await self._extract_visual_features(video)
                    features.append(feature)
                    
            except Exception as e:
                logger.error(f"[PATTERN_MINER] Error extracting features for video {video['video_id']}: {e}")
                continue
        
        return embeddings, features
    
    async def _extract_visual_features(self, video: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract visual features from video metadata
        """
        # This would analyze the actual thumbnail image
        # For now, we'll extract features from metadata
        
        features = {
            "title_length": len(video["title"]),
            "has_numbers": any(char.isdigit() for char in video["title"]),
            "has_question": "?" in video["title"],
            "has_exclamation": "!" in video["title"],
            "caps_percentage": sum(1 for c in video["title"] if c.isupper()) / len(video["title"]) if video["title"] else 0,
            "word_count": len(video["title"].split()),
            "engagement_rate": video["engagement_rate"],
            "duration_category": self._categorize_duration(video["duration"]),
            "tag_count": len(video["tags"]),
            "has_face": self._detect_face_keywords(video["title"]),
            "has_emoji": any(ord(char) > 127 for char in video["title"]),
            "success_indicators": self._extract_success_indicators(video)
        }
        
        return features
    
    def _categorize_duration(self, duration: str) -> str:
        """
        Categorize video duration
        """
        # Parse ISO 8601 duration (PT4M13S)
        if not duration.startswith("PT"):
            return "unknown"
        
        duration = duration[2:]  # Remove PT
        
        if "H" in duration:
            return "long"
        elif "M" in duration:
            minutes = int(duration.split("M")[0])
            if minutes <= 5:
                return "short"
            elif minutes <= 15:
                return "medium"
            else:
                return "long"
        else:
            return "short"
    
    def _detect_face_keywords(self, title: str) -> bool:
        """
        Detect if title suggests face/people content
        """
        face_keywords = ["face", "reaction", "interview", "meet", "talk", "chat", "vlog", "myself", "i", "me"]
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in face_keywords)
    
    def _extract_success_indicators(self, video: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract indicators of video success
        """
        return {
            "high_engagement": video["engagement_rate"] > 0.05,
            "viral_potential": video["views_per_hour"] > 1000,
            "trending_keywords": self._detect_trending_keywords(video["title"]),
            "clickbait_score": self._calculate_clickbait_score(video["title"])
        }
    
    def _detect_trending_keywords(self, title: str) -> List[str]:
        """
        Detect trending keywords in title
        """
        trending_keywords = ["new", "latest", "2024", "2025", "breaking", "exclusive", "first", "never", "secret", "hidden"]
        title_lower = title.lower()
        return [kw for kw in trending_keywords if kw in title_lower]
    
    def _calculate_clickbait_score(self, title: str) -> float:
        """
        Calculate clickbait score based on title characteristics
        """
        score = 0
        
        # Caps percentage
        caps_pct = sum(1 for c in title if c.isupper()) / len(title) if title else 0
        score += caps_pct * 0.3
        
        # Question marks
        score += title.count("?") * 0.1
        
        # Exclamation marks
        score += title.count("!") * 0.1
        
        # Numbers
        score += min(title.count("0") + title.count("1") + title.count("2") + title.count("3") + title.count("4") + title.count("5") + title.count("6") + title.count("7") + title.count("8") + title.count("9"), 3) * 0.1
        
        # Power words
        power_words = ["insane", "shocking", "secret", "exposed", "never", "ultimate", "best", "worst", "amazing", "incredible"]
        title_lower = title.lower()
        score += sum(0.2 for word in power_words if word in title_lower)
        
        return min(score, 1.0)
    
    async def _cluster_embeddings(self, embeddings: List[List[float]], videos: List[Dict[str, Any]], niche: Optional[str]) -> List[VisualPattern]:
        """
        Cluster embeddings to find visual patterns
        """
        if len(embeddings) < 10:
            return []
        
        embeddings_array = np.array(embeddings)
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings_array)
        
        # Determine optimal number of clusters
        n_clusters = min(8, max(3, len(embeddings) // 20))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_normalized)
        
        # Analyze each cluster
        patterns = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) < 3:  # Skip small clusters
                continue
            
            cluster_videos = [videos[i] for i in cluster_indices]
            cluster_embeddings = embeddings_normalized[cluster_indices]
            
            # Calculate cluster statistics
            avg_views_per_hour = np.mean([v["views_per_hour"] for v in cluster_videos])
            success_rate = len([v for v in cluster_videos if v["views_per_hour"] > np.percentile([v["views_per_hour"] for v in videos], 75)]) / len(cluster_videos)
            
            # Find common features
            common_features = self._analyze_cluster_features(cluster_videos)
            
            # Get example thumbnails
            example_thumbnails = [v["thumbnail_url"] for v in sorted(cluster_videos, key=lambda x: x["views_per_hour"], reverse=True)[:3]]
            
            pattern = VisualPattern(
                pattern_id=f"{niche or 'global'}_pattern_{cluster_id}",
                niche=niche or "global",
                cluster_center=kmeans.cluster_centers_[cluster_id].tolist(),
                success_rate=success_rate,
                avg_views_per_hour=avg_views_per_hour,
                common_features=common_features,
                thumbnail_examples=example_thumbnails,
                pattern_description=self._generate_pattern_description(common_features, cluster_videos)
            )
            
            patterns.append(pattern)
        
        # Sort patterns by success rate
        patterns.sort(key=lambda p: p.success_rate, reverse=True)
        
        return patterns
    
    def _analyze_cluster_features(self, cluster_videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze common features in a cluster
        """
        features = {
            "common_words": self._find_common_words([v["title"] for v in cluster_videos]),
            "avg_title_length": np.mean([len(v["title"]) for v in cluster_videos]),
            "common_tags": self._find_common_tags([v["tags"] for v in cluster_videos]),
            "duration_distribution": Counter([self._categorize_duration(v["duration"]) for v in cluster_videos]),
            "engagement_patterns": {
                "avg_engagement": np.mean([v["engagement_rate"] for v in cluster_videos]),
                "high_engagement_rate": len([v for v in cluster_videos if v["engagement_rate"] > 0.05]) / len(cluster_videos)
            }
        }
        
        return features
    
    def _find_common_words(self, titles: List[str]) -> List[Tuple[str, int]]:
        """
        Find most common words across titles
        """
        word_count = Counter()
        for title in titles:
            words = title.lower().split()
            word_count.update(words)
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
        
        filtered_words = [(word, count) for word, count in word_count.most_common(20) if word not in stop_words and len(word) > 2]
        
        return filtered_words[:10]
    
    def _find_common_tags(self, tags_lists: List[List[str]]) -> List[Tuple[str, int]]:
        """
        Find most common tags across videos
        """
        tag_count = Counter()
        for tags in tags_lists:
            tag_count.update(tags)
        
        return tag_count.most_common(10)
    
    def _generate_pattern_description(self, features: Dict[str, Any], videos: List[Dict[str, Any]]) -> str:
        """
        Generate human-readable description of a pattern
        """
        descriptions = []
        
        # Title characteristics
        avg_length = features["avg_title_length"]
        if avg_length < 50:
            descriptions.append("Short, punchy titles")
        elif avg_length > 80:
            descriptions.append("Detailed, descriptive titles")
        
        # Common words
        common_words = [word for word, count in features["common_words"][:3]]
        if common_words:
            descriptions.append(f"Common words: {', '.join(common_words)}")
        
        # Engagement
        avg_engagement = features["engagement_patterns"]["avg_engagement"]
        if avg_engagement > 0.05:
            descriptions.append("High engagement rate")
        
        # Duration
        duration_dist = features["duration_distribution"]
        most_common_duration = max(duration_dist, key=duration_dist.get)
        descriptions.append(f"Mostly {most_common_duration} videos")
        
        return "; ".join(descriptions)
    
    async def _mine_feature_patterns(self, features: List[Dict[str, Any]], videos: List[Dict[str, Any]]) -> List[FeaturePattern]:
        """
        Mine patterns from individual features
        """
        feature_patterns = []
        
        # Analyze each feature type
        feature_types = {
            "title_length": "text",
            "has_numbers": "text",
            "has_question": "text",
            "has_exclamation": "text",
            "caps_percentage": "text",
            "word_count": "text",
            "engagement_rate": "performance",
            "duration_category": "content",
            "tag_count": "metadata",
            "has_face": "visual",
            "has_emoji": "text"
        }
        
        for feature_name, feature_type in feature_types.items():
            pattern = await self._analyze_feature_pattern(feature_name, feature_type, features, videos)
            if pattern:
                feature_patterns.append(pattern)
        
        return feature_patterns
    
    async def _analyze_feature_pattern(self, feature_name: str, feature_type: str, features: List[Dict[str, Any]], videos: List[Dict[str, Any]]) -> Optional[FeaturePattern]:
        """
        Analyze patterns for a specific feature
        """
        feature_values = [f[feature_name] for f in features]
        
        if not feature_values:
            return None
        
        # Calculate correlation with success
        success_metrics = [v["views_per_hour"] for v in videos]
        
        # Find optimal threshold
        thresholds = np.percentile(feature_values, [25, 50, 75])
        
        best_threshold = None
        best_impact = 0
        
        for threshold in thresholds:
            high_performers = [success for i, success in enumerate(success_metrics) if feature_values[i] >= threshold]
            low_performers = [success for i, success in enumerate(success_metrics) if feature_values[i] < threshold]
            
            if len(high_performers) > 0 and len(low_performers) > 0:
                impact = np.mean(high_performers) - np.mean(low_performers)
                if impact > best_impact:
                    best_impact = impact
                    best_threshold = threshold
        
        if best_threshold is None or best_impact < 100:  # Minimum impact threshold
            return None
        
        # Get examples
        examples = [videos[i]["title"] for i, val in enumerate(feature_values) if val >= best_threshold][:5]
        
        return FeaturePattern(
            feature_name=feature_name,
            feature_type=feature_type,
            success_threshold=best_threshold,
            impact_score=best_impact,
            examples=examples
        )
    
    async def _store_patterns(self, visual_patterns: List[VisualPattern], feature_patterns: List[FeaturePattern], niche: Optional[str]):
        """
        Store discovered patterns in database
        """
        try:
            # Store visual patterns
            pattern_records = []
            for pattern in visual_patterns:
                record = {
                    "pattern_id": pattern.pattern_id,
                    "niche": pattern.niche,
                    "cluster_center": pattern.cluster_center,
                    "success_rate": pattern.success_rate,
                    "avg_views_per_hour": pattern.avg_views_per_hour,
                    "common_features": pattern.common_features,
                    "thumbnail_examples": pattern.thumbnail_examples,
                    "pattern_description": pattern.pattern_description,
                    "created_at": "now()"
                }
                pattern_records.append(record)
            
            if pattern_records:
                self.supabase.table("visual_patterns").upsert(pattern_records, on_conflict="pattern_id").execute()
            
            # Store feature patterns
            feature_records = []
            for pattern in feature_patterns:
                record = {
                    "feature_name": pattern.feature_name,
                    "feature_type": pattern.feature_type,
                    "success_threshold": pattern.success_threshold,
                    "impact_score": pattern.impact_score,
                    "examples": pattern.examples,
                    "niche": niche or "global",
                    "created_at": "now()"
                }
                feature_records.append(record)
            
            if feature_records:
                self.supabase.table("feature_patterns").upsert(feature_records, on_conflict="feature_name,niche").execute()
            
            logger.info(f"[PATTERN_MINER] Stored {len(pattern_records)} visual patterns and {len(feature_records)} feature patterns")
            
        except Exception as e:
            logger.error(f"[PATTERN_MINER] Error storing patterns: {e}")

# Example usage
async def mine_patterns():
    """
    Main function to mine patterns from YouTube data
    """
    from supabase import create_client
    import os
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not all([supabase_url, supabase_key]):
        logger.error("[PATTERN_MINER] Missing Supabase credentials")
        return
    
    # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
    supabase = create_client(supabase_url, supabase_key)
    miner = PatternMiner(supabase)
    
    # Mine patterns for each niche
    niches = ["tech", "gaming", "education", "entertainment", "people"]
    
    for niche in niches:
        logger.info(f"[PATTERN_MINER] Mining patterns for {niche}")
        patterns = await miner.mine_patterns(niche)
        logger.info(f"[PATTERN_MINER] Completed {niche}: {len(patterns.get('visual_patterns', []))} visual patterns, {len(patterns.get('feature_patterns', []))} feature patterns")
    
    # Mine global patterns
    logger.info("[PATTERN_MINER] Mining global patterns")
    global_patterns = await miner.mine_patterns()
    logger.info(f"[PATTERN_MINER] Completed global: {len(global_patterns.get('visual_patterns', []))} visual patterns, {len(global_patterns.get('feature_patterns', []))} feature patterns")

if __name__ == "__main__":
    import asyncio
    asyncio.run(mine_patterns())
