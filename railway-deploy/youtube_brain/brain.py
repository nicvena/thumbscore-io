"""
YouTube Intelligence Brain - Unified Interface
Main interface that coordinates all brain components for intelligent thumbnail scoring
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Import brain components
from .data_collector import YouTubeDataCollector
from .pattern_miner import PatternMiner, VisualPattern, FeaturePattern
from .niche_models import NicheModelTrainer, PredictionResult
from .trend_detector import TrendDetector, VisualTrend, TrendAlert
from .insights_engine import InsightsEngine, CreatorInsight

logger = logging.getLogger(__name__)

@dataclass
class BrainScore:
    thumbnail_id: str
    niche: str
    brain_weighted_score: float
    confidence: float
    pattern_matches: List[Dict[str, Any]]
    trend_alignment: float
    creator_insights: Optional[Dict[str, Any]]
    model_predictions: Optional[PredictionResult]
    explanations: List[str]

@dataclass
class BrainStatus:
    data_collector_ready: bool
    pattern_miner_ready: bool
    niche_models_ready: bool
    trend_detector_ready: bool
    insights_engine_ready: bool
    last_data_update: Optional[datetime]
    total_patterns: int
    total_trends: int
    trained_niches: List[str]

class YouTubeBrain:
    """
    Main YouTube Intelligence Brain that coordinates all components
    """
    
    def __init__(self, supabase_client, youtube_api_key: str):
        self.supabase = supabase_client
        self.youtube_api_key = youtube_api_key
        
        # Initialize components
        self.data_collector = YouTubeDataCollector(youtube_api_key, supabase_client)
        self.pattern_miner = PatternMiner(supabase_client)
        self.niche_models = NicheModelTrainer(supabase_client)
        self.trend_detector = TrendDetector(supabase_client)
        self.insights_engine = InsightsEngine(supabase_client)
        
        # Brain state
        self.is_initialized = False
        self.last_update = None
    
    async def initialize(self) -> BrainStatus:
        """
        Initialize the YouTube Brain by training all components
        """
        logger.info("[BRAIN] Initializing YouTube Intelligence Brain...")
        
        try:
            # Step 1: Collect fresh data
            logger.info("[BRAIN] Step 1: Collecting YouTube data...")
            await self.data_collector.collect_trending_data(max_videos_per_niche=200)
            await self.data_collector.store_collected_data({})
            
            # Step 2: Mine patterns
            logger.info("[BRAIN] Step 2: Mining visual patterns...")
            await self.pattern_miner.mine_patterns()
            
            # Step 3: Train niche models
            logger.info("[BRAIN] Step 3: Training niche models...")
            await self.niche_models.train_niche_models()
            
            # Step 4: Detect trends
            logger.info("[BRAIN] Step 4: Detecting trends...")
            await self.trend_detector.detect_trends()
            
            # Step 5: Generate creator insights
            logger.info("[BRAIN] Step 5: Generating creator insights...")
            await self.insights_engine.generate_creator_insights()
            
            self.is_initialized = True
            self.last_update = datetime.now()
            
            logger.info("[BRAIN] ✅ YouTube Intelligence Brain initialized successfully!")
            
            return await self.get_status()
            
        except Exception as e:
            logger.error(f"[BRAIN] ❌ Failed to initialize brain: {e}")
            return await self.get_status()
    
    async def score_thumbnail(self, embedding: List[float], niche: str, features: Dict[str, Any], channel_id: Optional[str] = None) -> BrainScore:
        """
        Score a thumbnail using the full YouTube Intelligence Brain
        """
        if not self.is_initialized:
            logger.warning("[BRAIN] Brain not initialized, using fallback scoring")
            return self._fallback_score(embedding, niche, features)
        
        try:
            logger.info(f"[BRAIN] Scoring thumbnail for niche: {niche}")
            
            # Step 1: Pattern matching
            pattern_matches = await self._find_pattern_matches(embedding, niche)
            
            # Step 2: Trend alignment
            trend_alignment = await self._calculate_trend_alignment(features, niche)
            
            # Step 3: Niche model prediction
            model_predictions = await self._get_model_predictions(embedding, features, niche)
            
            # Step 4: Creator insights (if channel_id provided)
            creator_insights = None
            if channel_id:
                creator_insights = await self.insights_engine.get_creator_insights(channel_id)
            
            # Step 5: Calculate brain-weighted score
            brain_score = self._calculate_brain_score(
                pattern_matches, trend_alignment, model_predictions, creator_insights
            )
            
            # Step 6: Generate explanations
            explanations = self._generate_explanations(
                pattern_matches, trend_alignment, model_predictions, creator_insights
            )
            
            return BrainScore(
                thumbnail_id=features.get("thumbnail_id", "unknown"),
                niche=niche,
                brain_weighted_score=brain_score,
                confidence=self._calculate_confidence(pattern_matches, model_predictions),
                pattern_matches=pattern_matches,
                trend_alignment=trend_alignment,
                creator_insights=creator_insights.__dict__ if creator_insights else None,
                model_predictions=model_predictions,
                explanations=explanations
            )
            
        except Exception as e:
            logger.error(f"[BRAIN] Error scoring thumbnail: {e}")
            return self._fallback_score(embedding, niche, features)
    
    async def _find_pattern_matches(self, embedding: List[float], niche: str) -> List[Dict[str, Any]]:
        """
        Find matching visual patterns for the thumbnail
        """
        try:
            # Get visual patterns for the niche
            query = self.supabase.table("visual_patterns").select("*").eq("niche", niche)
            result = query.execute()
            patterns = result.data
            
            matches = []
            for pattern in patterns:
                # Calculate similarity to pattern center
                pattern_center = np.array(pattern["cluster_center"])
                thumbnail_embedding = np.array(embedding)
                
                # Cosine similarity
                similarity = np.dot(pattern_center, thumbnail_embedding) / (
                    np.linalg.norm(pattern_center) * np.linalg.norm(thumbnail_embedding)
                )
                
                if similarity > 0.7:  # High similarity threshold
                    matches.append({
                        "pattern_id": pattern["pattern_id"],
                        "pattern_description": pattern["pattern_description"],
                        "similarity": float(similarity),
                        "success_rate": pattern["success_rate"],
                        "avg_views_per_hour": pattern["avg_views_per_hour"]
                    })
            
            # Sort by similarity
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            return matches[:3]  # Top 3 matches
            
        except Exception as e:
            logger.error(f"[BRAIN] Error finding pattern matches: {e}")
            return []
    
    async def _calculate_trend_alignment(self, features: Dict[str, Any], niche: str) -> float:
        """
        Calculate how well the thumbnail aligns with current trends
        """
        try:
            # Get current trends for the niche
            query = self.supabase.table("visual_trends").select("*").eq("niche", niche)
            query = query.order("trend_strength", desc=True).limit(10)
            result = query.execute()
            trends = result.data
            
            if not trends:
                return 0.5  # Neutral alignment
            
            alignment_score = 0.0
            total_weight = 0.0
            
            for trend in trends:
                # Calculate alignment based on trend type and features
                trend_alignment = self._calculate_trend_feature_alignment(trend, features)
                
                # Weight by trend strength
                weight = trend["trend_strength"]
                alignment_score += trend_alignment * weight
                total_weight += weight
            
            return alignment_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"[BRAIN] Error calculating trend alignment: {e}")
            return 0.5
    
    def _calculate_trend_feature_alignment(self, trend: Dict[str, Any], features: Dict[str, Any]) -> float:
        """
        Calculate alignment between a specific trend and thumbnail features
        """
        trend_type = trend["trend_type"]
        
        if trend_type == "text":
            # Analyze text features
            title = features.get("title", "")
            if not title:
                return 0.5
            
            # Check for trend-specific text patterns
            if "caps_heavy" in trend["trend_description"]:
                caps_pct = sum(1 for c in title if c.isupper()) / len(title)
                return min(caps_pct * 2, 1.0)
            
            elif "question" in trend["trend_description"]:
                return 1.0 if "?" in title else 0.0
            
            elif "exclamation" in trend["trend_description"]:
                return 1.0 if "!" in title else 0.0
        
        elif trend_type == "color":
            # Analyze color features (simplified)
            # In a real implementation, this would analyze actual image colors
            return 0.5  # Placeholder
        
        elif trend_type == "composition":
            # Analyze composition features
            title = features.get("title", "")
            if "face" in trend["trend_description"]:
                face_keywords = ["face", "reaction", "interview", "meet", "talk"]
                return 1.0 if any(keyword in title.lower() for keyword in face_keywords) else 0.0
        
        return 0.5  # Default neutral alignment
    
    async def _get_model_predictions(self, embedding: List[float], features: Dict[str, Any], niche: str) -> Optional[PredictionResult]:
        """
        Get predictions from niche models
        """
        try:
            # Extract additional features
            additional_features = self._extract_additional_features(features)
            
            # Get model predictions
            predictions = await self.niche_models.predict_thumbnail_performance(
                embedding, additional_features, niche
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"[BRAIN] Error getting model predictions: {e}")
            return None
    
    def _extract_additional_features(self, features: Dict[str, Any]) -> List[float]:
        """
        Extract additional features for model prediction
        """
        additional_features = []
        
        # Title features
        title = features.get("title", "")
        additional_features.extend([
            len(title),  # Title length
            title.count("?"),  # Question marks
            title.count("!"),  # Exclamation marks
            sum(1 for c in title if c.isupper()) / len(title) if title else 0,  # Caps percentage
            len(title.split()),  # Word count
            sum(1 for c in title if c.isdigit()),  # Number count
        ])
        
        # Content features
        additional_features.extend([
            len(features.get("tags", [])),  # Tag count
            features.get("like_rate", 0),  # Like rate
            features.get("comment_rate", 0),  # Comment rate
        ])
        
        # Duration features
        duration = features.get("duration", "PT0S")
        duration_seconds = self._parse_duration(duration)
        additional_features.extend([
            duration_seconds,
            duration_seconds / 3600,  # Hours
            1 if duration_seconds < 300 else 0,  # Short video indicator
            1 if duration_seconds > 1800 else 0,  # Long video indicator
        ])
        
        # Temporal features
        hours_ago = features.get("hours_ago", 24)
        additional_features.extend([
            hours_ago,
            np.log(hours_ago + 1),  # Log hours ago
        ])
        
        # Performance features
        additional_features.extend([
            features.get("views_per_hour", 100),
            features.get("engagement_rate", 0.03),
        ])
        
        return additional_features
    
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
    
    def _calculate_brain_score(self, pattern_matches: List[Dict[str, Any]], trend_alignment: float, model_predictions: Optional[PredictionResult], creator_insights: Optional[CreatorInsight]) -> float:
        """
        Calculate the final brain-weighted score
        """
        # Base score from model predictions
        base_score = 0.5
        if model_predictions:
            base_score = model_predictions.predicted_ctr
        
        # Pattern bonus
        pattern_bonus = 0.0
        if pattern_matches:
            avg_success_rate = np.mean([match["success_rate"] for match in pattern_matches])
            pattern_bonus = avg_success_rate * 0.2  # Up to 20% bonus
        
        # Trend alignment bonus
        trend_bonus = (trend_alignment - 0.5) * 0.3  # Up to 15% bonus/penalty
        
        # Creator insights bonus
        creator_bonus = 0.0
        if creator_insights and creator_insights.performance_trends:
            if creator_insights.performance_trends.get("views_trend") == "up":
                creator_bonus = 0.1  # 10% bonus for improving creators
        
        # Calculate final score
        final_score = base_score + pattern_bonus + trend_bonus + creator_bonus
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))
    
    def _calculate_confidence(self, pattern_matches: List[Dict[str, Any]], model_predictions: Optional[PredictionResult]) -> float:
        """
        Calculate confidence in the brain score
        """
        confidence = 0.5  # Base confidence
        
        # Pattern confidence
        if pattern_matches:
            avg_similarity = np.mean([match["similarity"] for match in pattern_matches])
            confidence += avg_similarity * 0.3
        
        # Model confidence
        if model_predictions:
            confidence += model_predictions.confidence * 0.2
        
        return min(confidence, 1.0)
    
    def _generate_explanations(self, pattern_matches: List[Dict[str, Any]], trend_alignment: float, model_predictions: Optional[PredictionResult], creator_insights: Optional[CreatorInsight]) -> List[str]:
        """
        Generate human-readable explanations for the score
        """
        explanations = []
        
        # Pattern explanations
        if pattern_matches:
            best_match = pattern_matches[0]
            explanations.append(f"🎯 Matches successful pattern: {best_match['pattern_description']}")
            explanations.append(f"📈 Pattern success rate: {best_match['success_rate']:.1%}")
        
        # Trend explanations
        if trend_alignment > 0.7:
            explanations.append("🚀 Aligns well with current trends")
        elif trend_alignment < 0.3:
            explanations.append("⚠️ Doesn't align with current trends")
        
        # Model explanations
        if model_predictions and model_predictions.model_explanations:
            explanations.extend(model_predictions.model_explanations[:2])
        
        # Creator insights
        if creator_insights and creator_insights.personalized_recommendations:
            explanations.append(f"💡 Personalized tip: {creator_insights.personalized_recommendations[0]}")
        
        return explanations
    
    def _fallback_score(self, embedding: List[float], niche: str, features: Dict[str, Any]) -> BrainScore:
        """
        Fallback scoring when brain is not initialized
        """
        return BrainScore(
            thumbnail_id=features.get("thumbnail_id", "unknown"),
            niche=niche,
            brain_weighted_score=0.5,
            confidence=0.3,
            pattern_matches=[],
            trend_alignment=0.5,
            creator_insights=None,
            model_predictions=None,
            explanations=["Brain not initialized - using fallback scoring"]
        )
    
    async def get_trending_patterns(self, niche: str) -> List[VisualTrend]:
        """
        Get trending patterns for a specific niche
        """
        try:
            query = self.supabase.table("visual_trends").select("*").eq("niche", niche)
            query = query.order("trend_strength", desc=True).limit(10)
            result = query.execute()
            
            trends = []
            for trend_data in result.data:
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
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"[BRAIN] Error getting trending patterns: {e}")
            return []
    
    async def get_creator_insights(self, channel_id: str) -> Optional[CreatorInsight]:
        """
        Get insights for a specific creator
        """
        return await self.insights_engine.get_creator_insights(channel_id)
    
    async def get_status(self) -> BrainStatus:
        """
        Get the current status of the YouTube Brain
        """
        try:
            # Check data collector status
            data_collector_ready = True  # Assume ready if no errors
            
            # Check pattern miner status
            pattern_result = self.supabase.table("visual_patterns").select("pattern_id").limit(1).execute()
            pattern_miner_ready = len(pattern_result.data) > 0
            
            # Check niche models status
            model_result = self.supabase.table("model_performance").select("niche").limit(1).execute()
            niche_models_ready = len(model_result.data) > 0
            
            # Check trend detector status
            trend_result = self.supabase.table("visual_trends").select("trend_id").limit(1).execute()
            trend_detector_ready = len(trend_result.data) > 0
            
            # Check insights engine status
            insights_result = self.supabase.table("creator_insights").select("channel_id").limit(1).execute()
            insights_engine_ready = len(insights_result.data) > 0
            
            # Get counts
            total_patterns = len(self.supabase.table("visual_patterns").select("pattern_id").execute().data)
            total_trends = len(self.supabase.table("visual_trends").select("trend_id").execute().data)
            
            # Get trained niches
            trained_niches = [row["niche"] for row in self.supabase.table("model_performance").select("niche").execute().data]
            
            return BrainStatus(
                data_collector_ready=data_collector_ready,
                pattern_miner_ready=pattern_miner_ready,
                niche_models_ready=niche_models_ready,
                trend_detector_ready=trend_detector_ready,
                insights_engine_ready=insights_engine_ready,
                last_data_update=self.last_update,
                total_patterns=total_patterns,
                total_trends=total_trends,
                trained_niches=trained_niches
            )
            
        except Exception as e:
            logger.error(f"[BRAIN] Error getting status: {e}")
            return BrainStatus(
                data_collector_ready=False,
                pattern_miner_ready=False,
                niche_models_ready=False,
                trend_detector_ready=False,
                insights_engine_ready=False,
                last_data_update=None,
                total_patterns=0,
                total_trends=0,
                trained_niches=[]
            )
    
    async def refresh_brain(self) -> BrainStatus:
        """
        Refresh the brain with new data and retrain models
        """
        logger.info("[BRAIN] Refreshing YouTube Intelligence Brain...")
        
        try:
            # Collect new data
            await self.data_collector.collect_trending_data(max_videos_per_niche=200)
            
            # Retrain components
            await self.pattern_miner.mine_patterns()
            await self.niche_models.train_niche_models()
            await self.trend_detector.detect_trends()
            await self.insights_engine.generate_creator_insights()
            
            self.last_update = datetime.now()
            
            logger.info("[BRAIN] ✅ Brain refresh completed!")
            return await self.get_status()
            
        except Exception as e:
            logger.error(f"[BRAIN] ❌ Error refreshing brain: {e}")
            return await self.get_status()

# Example usage
async def initialize_brain():
    """
    Main function to initialize the YouTube Intelligence Brain
    """
    from supabase import create_client
    import os
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    
    if not all([supabase_url, supabase_key, youtube_key]):
        logger.error("[BRAIN] Missing required environment variables")
        return
    
    # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
    supabase = create_client(supabase_url, supabase_key)
    brain = YouTubeBrain(supabase, youtube_key)
    
    # Initialize the brain
    status = await brain.initialize()
    
    logger.info(f"[BRAIN] Brain status: {status}")
    logger.info(f"[BRAIN] Patterns: {status.total_patterns}, Trends: {status.total_trends}")
    logger.info(f"[BRAIN] Trained niches: {status.trained_niches}")

if __name__ == "__main__":
    asyncio.run(initialize_brain())
