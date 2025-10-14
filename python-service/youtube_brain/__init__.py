"""
YouTube Intelligence Brain Module

A comprehensive AI system that learns from real YouTube data to understand
what makes thumbnails successful. Provides intelligent, data-driven scoring
and personalized recommendations for YouTube creators.

Components:
- data_collector: Collects trending YouTube videos and metadata
- pattern_miner: Discovers visual patterns in successful thumbnails  
- niche_models: Trains ML models to predict thumbnail performance
- trend_detector: Detects emerging visual trends
- insights_engine: Generates personalized creator insights
- brain: Unified interface coordinating all components

Usage:
    from youtube_brain.brain import YouTubeBrain
    
    brain = YouTubeBrain(supabase_client, youtube_api_key)
    await brain.initialize()
    
    score = await brain.score_thumbnail(embedding, niche, features)
"""

from .brain import YouTubeBrain, BrainScore, BrainStatus
from .data_collector import YouTubeDataCollector, VideoData
from .pattern_miner import PatternMiner, VisualPattern, FeaturePattern
from .niche_models import NicheModelTrainer, PredictionResult, ModelPerformance
from .trend_detector import TrendDetector, VisualTrend, TrendAlert
from .insights_engine import InsightsEngine, CreatorInsight, PerformanceInsight

__version__ = "1.0.0"
__author__ = "Thumbscore.io Team"

__all__ = [
    "YouTubeBrain",
    "BrainScore", 
    "BrainStatus",
    "YouTubeDataCollector",
    "VideoData",
    "PatternMiner",
    "VisualPattern",
    "FeaturePattern", 
    "NicheModelTrainer",
    "PredictionResult",
    "ModelPerformance",
    "TrendDetector",
    "VisualTrend",
    "TrendAlert",
    "InsightsEngine",
    "CreatorInsight",
    "PerformanceInsight"
]
