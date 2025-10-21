"""
YouTube Intelligence Brain - Niche Models
Trains small regressors per niche to predict CTR and views/hour
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
import json
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    model_name: str
    niche: str
    r2_score: float
    mse: float
    feature_importance: Dict[str, float]
    training_samples: int
    validation_samples: int

@dataclass
class PredictionResult:
    predicted_ctr: float
    predicted_views_per_hour: float
    confidence: float
    feature_contributions: Dict[str, float]
    model_explanations: List[str]

class NicheModelTrainer:
    """
    Trains and manages niche-specific models for thumbnail performance prediction
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_performance = {}
    
    async def train_niche_models(self, niche: Optional[str] = None) -> Dict[str, ModelPerformance]:
        """
        Train models for specific niche or all niches
        """
        logger.info(f"[NICHE_MODELS] Training models for niche: {niche or 'all'}")
        
        if niche:
            niches = [niche]
        else:
            niches = ["tech", "gaming", "education", "entertainment", "people", "business", "music", "sports", "news", "comedy", "howto"]
        
        performance_results = {}
        
        for niche_name in niches:
            try:
                logger.info(f"[NICHE_MODELS] Training models for {niche_name}")
                performance = await self._train_niche_model(niche_name)
                performance_results[niche_name] = performance
                
            except Exception as e:
                logger.error(f"[NICHE_MODELS] Error training {niche_name} model: {e}")
                continue
        
        # Store model performance
        await self._store_model_performance(performance_results)
        
        logger.info(f"[NICHE_MODELS] Completed training for {len(performance_results)} niches")
        return performance_results
    
    async def _train_niche_model(self, niche: str) -> ModelPerformance:
        """
        Train models for a specific niche
        """
        # Get training data
        X, y_ctr, y_views = await self._prepare_training_data(niche)
        
        if len(X) < 50:  # Need minimum samples
            logger.warning(f"[NICHE_MODELS] Insufficient data for {niche}: {len(X)} samples")
            return ModelPerformance(
                model_name="insufficient_data",
                niche=niche,
                r2_score=0.0,
                mse=0.0,
                feature_importance={},
                training_samples=len(X),
                validation_samples=0
            )
        
        # Split data
        X_train, X_val, y_ctr_train, y_ctr_val, y_views_train, y_views_val = train_test_split(
            X, y_ctr, y_views, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train CTR prediction model
        ctr_model = self._train_ctr_model(X_train_scaled, y_ctr_train)
        ctr_score = ctr_model.score(X_val_scaled, y_ctr_val)
        ctr_predictions = ctr_model.predict(X_val_scaled)
        ctr_mse = mean_squared_error(y_ctr_val, ctr_predictions)
        
        # Train views prediction model
        views_model = self._train_views_model(X_train_scaled, y_views_train)
        views_score = views_model.score(X_val_scaled, y_views_val)
        views_predictions = views_model.predict(X_val_scaled)
        views_mse = mean_squared_error(y_views_val, views_predictions)
        
        # Get feature importance
        feature_importance = self._get_feature_importance(ctr_model, views_model)
        
        # Store models
        model_key = f"{niche}_models"
        self.models[model_key] = {
            "ctr_model": ctr_model,
            "views_model": views_model
        }
        self.scalers[model_key] = scaler
        
        # Save models to disk
        await self._save_models(niche, ctr_model, views_model, scaler)
        
        logger.info(f"[NICHE_MODELS] {niche} - CTR R²: {ctr_score:.3f}, Views R²: {views_score:.3f}")
        
        return ModelPerformance(
            model_name=f"{niche}_ensemble",
            niche=niche,
            r2_score=(ctr_score + views_score) / 2,  # Average R²
            mse=(ctr_mse + views_mse) / 2,  # Average MSE
            feature_importance=feature_importance,
            training_samples=len(X_train),
            validation_samples=len(X_val)
        )
    
    async def _prepare_training_data(self, niche: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for niche model
        """
        try:
            # Get videos with embeddings and performance data
            query = self.supabase.table("youtube_videos").select("*").eq("niche", niche)
            result = query.execute()
            videos = result.data
            
            if not videos:
                logger.warning(f"[NICHE_MODELS] No videos found for {niche}")
                return np.array([]), np.array([]), np.array([])
            
            # Extract features and targets
            features = []
            ctr_targets = []
            views_targets = []
            
            for video in videos:
                try:
                    # Get CLIP embedding
                    embedding_result = self.supabase.table("ref_thumbnails").select("embedding").eq("thumb_url", video["thumbnail_url"]).execute()
                    
                    if not embedding_result.data:
                        continue
                    
                    embedding = embedding_result.data[0]["embedding"]
                    
                    # Extract additional features
                    additional_features = self._extract_additional_features(video)
                    
                    # Combine embedding with additional features
                    combined_features = np.concatenate([embedding, additional_features])
                    features.append(combined_features)
                    
                    # Calculate CTR proxy (engagement rate)
                    ctr_proxy = video["engagement_rate"]
                    ctr_targets.append(ctr_proxy)
                    
                    # Views per hour as target
                    views_targets.append(video["views_per_hour"])
                    
                except Exception as e:
                    logger.error(f"[NICHE_MODELS] Error processing video {video['video_id']}: {e}")
                    continue
            
            if not features:
                logger.warning(f"[NICHE_MODELS] No valid features extracted for {niche}")
                return np.array([]), np.array([]), np.array([])
            
            X = np.array(features)
            y_ctr = np.array(ctr_targets)
            y_views = np.array(views_targets)
            
            logger.info(f"[NICHE_MODELS] Prepared {len(X)} samples for {niche}")
            return X, y_ctr, y_views
            
        except Exception as e:
            logger.error(f"[NICHE_MODELS] Error preparing training data for {niche}: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _extract_additional_features(self, video: Dict[str, Any]) -> np.ndarray:
        """
        Extract additional features from video metadata
        """
        features = []
        
        # Title features
        title = video["title"]
        features.extend([
            len(title),  # Title length
            title.count("?"),  # Question marks
            title.count("!"),  # Exclamation marks
            sum(1 for c in title if c.isupper()) / len(title) if title else 0,  # Caps percentage
            len(title.split()),  # Word count
            sum(1 for c in title if c.isdigit()),  # Number count
        ])
        
        # Content features
        features.extend([
            len(video["tags"]),  # Tag count
            video["like_count"] / max(video["view_count"], 1),  # Like rate
            video["comment_count"] / max(video["view_count"], 1),  # Comment rate
        ])
        
        # Duration features
        duration = video["duration"]
        duration_seconds = self._parse_duration(duration)
        features.extend([
            duration_seconds,
            duration_seconds / 3600,  # Hours
            1 if duration_seconds < 300 else 0,  # Short video indicator
            1 if duration_seconds > 1800 else 0,  # Long video indicator
        ])
        
        # Temporal features
        published_at = video["published_at"]
        hours_ago = (np.datetime64('now') - np.datetime64(published_at)) / np.timedelta64(1, 'h')
        features.extend([
            hours_ago,
            np.log(hours_ago + 1),  # Log hours ago
        ])
        
        # Channel features (if available)
        features.extend([
            video["view_count"] / max(hours_ago, 1),  # Views per hour
            video["engagement_rate"],
        ])
        
        return np.array(features)
    
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
    
    def _train_ctr_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train CTR prediction model
        """
        # Use ensemble of models for better performance
        models = [
            RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
            Ridge(alpha=1.0)
        ]
        
        # Train ensemble
        predictions = []
        for model in models:
            model.fit(X_train, y_train)
            pred = model.predict(X_train)
            predictions.append(pred)
        
        # Use best performing model
        best_model = None
        best_score = -np.inf
        
        for model in models:
            score = model.score(X_train, y_train)
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model
    
    def _train_views_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train views prediction model
        """
        # Use Random Forest for views prediction (handles non-linear relationships well)
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        return model
    
    def _get_feature_importance(self, ctr_model, views_model) -> Dict[str, float]:
        """
        Get feature importance from models
        """
        importance = {}
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        if hasattr(ctr_model, 'feature_importances_'):
            ctr_importance = ctr_model.feature_importances_
            for i, name in enumerate(feature_names):
                importance[f"ctr_{name}"] = float(ctr_importance[i])
        
        if hasattr(views_model, 'feature_importances_'):
            views_importance = views_model.feature_importances_
            for i, name in enumerate(feature_names):
                importance[f"views_{name}"] = float(views_importance[i])
        
        return importance
    
    def _get_feature_names(self) -> List[str]:
        """
        Get feature names for interpretability
        """
        if not self.feature_names:
            # CLIP embedding features (768 dimensions)
            clip_features = [f"clip_embedding_{i}" for i in range(768)]
            
            # Additional features
            additional_features = [
                "title_length", "question_marks", "exclamation_marks", "caps_percentage",
                "word_count", "number_count", "tag_count", "like_rate", "comment_rate",
                "duration_seconds", "duration_hours", "is_short_video", "is_long_video",
                "hours_ago", "log_hours_ago", "views_per_hour", "engagement_rate"
            ]
            
            self.feature_names = clip_features + additional_features
        
        return self.feature_names
    
    async def _save_models(self, niche: str, ctr_model, views_model, scaler):
        """
        Save trained models to disk
        """
        try:
            model_dir = f"models/{niche}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models
            joblib.dump(ctr_model, f"{model_dir}/ctr_model.pkl")
            joblib.dump(views_model, f"{model_dir}/views_model.pkl")
            joblib.dump(scaler, f"{model_dir}/scaler.pkl")
            
            logger.info(f"[NICHE_MODELS] Saved models for {niche}")
            
        except Exception as e:
            logger.error(f"[NICHE_MODELS] Error saving models for {niche}: {e}")
    
    async def _store_model_performance(self, performance_results: Dict[str, ModelPerformance]):
        """
        Store model performance metrics in database
        """
        try:
            records = []
            for niche, performance in performance_results.items():
                record = {
                    "niche": niche,
                    "model_name": performance.model_name,
                    "r2_score": performance.r2_score,
                    "mse": performance.mse,
                    "feature_importance": performance.feature_importance,
                    "training_samples": performance.training_samples,
                    "validation_samples": performance.validation_samples,
                    "created_at": "now()"
                }
                records.append(record)
            
            if records:
                self.supabase.table("model_performance").upsert(records, on_conflict="niche").execute()
                logger.info(f"[NICHE_MODELS] Stored performance metrics for {len(records)} models")
            
        except Exception as e:
            logger.error(f"[NICHE_MODELS] Error storing model performance: {e}")
    
    async def predict_thumbnail_performance(self, embedding: List[float], additional_features: List[float], niche: str) -> PredictionResult:
        """
        Predict thumbnail performance using trained models
        """
        try:
            model_key = f"{niche}_models"
            
            if model_key not in self.models:
                # Load models from disk
                await self._load_models(niche)
            
            if model_key not in self.models:
                logger.warning(f"[NICHE_MODELS] No trained model found for {niche}")
                return PredictionResult(
                    predicted_ctr=0.5,
                    predicted_views_per_hour=100,
                    confidence=0.0,
                    feature_contributions={},
                    model_explanations=["No trained model available"]
                )
            
            # Prepare features
            combined_features = np.concatenate([embedding, additional_features])
            features_scaled = self.scalers[model_key].transform([combined_features])
            
            # Make predictions
            ctr_model = self.models[model_key]["ctr_model"]
            views_model = self.models[model_key]["views_model"]
            
            predicted_ctr = float(ctr_model.predict(features_scaled)[0])
            predicted_views = float(views_model.predict(features_scaled)[0])
            
            # Calculate confidence based on model performance
            performance = self.model_performance.get(niche)
            confidence = performance.r2_score if performance else 0.5
            
            # Get feature contributions
            feature_contributions = self._get_feature_contributions(ctr_model, views_model, features_scaled[0])
            
            # Generate explanations
            explanations = self._generate_explanations(feature_contributions, predicted_ctr, predicted_views)
            
            return PredictionResult(
                predicted_ctr=predicted_ctr,
                predicted_views_per_hour=predicted_views,
                confidence=confidence,
                feature_contributions=feature_contributions,
                model_explanations=explanations
            )
            
        except Exception as e:
            logger.error(f"[NICHE_MODELS] Error predicting performance: {e}")
            return PredictionResult(
                predicted_ctr=0.5,
                predicted_views_per_hour=100,
                confidence=0.0,
                feature_contributions={},
                model_explanations=["Prediction error occurred"]
            )
    
    async def _load_models(self, niche: str):
        """
        Load trained models from disk
        """
        try:
            model_dir = f"models/{niche}"
            
            if not os.path.exists(model_dir):
                return
            
            ctr_model = joblib.load(f"{model_dir}/ctr_model.pkl")
            views_model = joblib.load(f"{model_dir}/views_model.pkl")
            scaler = joblib.load(f"{model_dir}/scaler.pkl")
            
            model_key = f"{niche}_models"
            self.models[model_key] = {
                "ctr_model": ctr_model,
                "views_model": views_model
            }
            self.scalers[model_key] = scaler
            
            logger.info(f"[NICHE_MODELS] Loaded models for {niche}")
            
        except Exception as e:
            logger.error(f"[NICHE_MODELS] Error loading models for {niche}: {e}")
    
    def _get_feature_contributions(self, ctr_model, views_model, features_scaled) -> Dict[str, float]:
        """
        Get feature contributions for interpretability
        """
        contributions = {}
        
        if hasattr(ctr_model, 'feature_importances_'):
            feature_names = self._get_feature_names()
            for i, name in enumerate(feature_names):
                contributions[f"ctr_{name}"] = float(ctr_model.feature_importances_[i] * features_scaled[i])
        
        if hasattr(views_model, 'feature_importances_'):
            feature_names = self._get_feature_names()
            for i, name in enumerate(feature_names):
                contributions[f"views_{name}"] = float(views_model.feature_importances_[i] * features_scaled[i])
        
        return contributions
    
    def _generate_explanations(self, contributions: Dict[str, float], predicted_ctr: float, predicted_views: float) -> List[str]:
        """
        Generate human-readable explanations
        """
        explanations = []
        
        # Top positive contributors
        positive_contributors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for feature, contribution in positive_contributors:
            if contribution > 0.01:  # Significant contribution
                explanations.append(f"Strong {feature} contributes +{contribution:.3f} to performance")
        
        # Performance interpretation
        if predicted_ctr > 0.05:
            explanations.append("High predicted engagement rate")
        elif predicted_ctr < 0.02:
            explanations.append("Low predicted engagement rate")
        
        if predicted_views > 1000:
            explanations.append("High predicted view velocity")
        elif predicted_views < 100:
            explanations.append("Low predicted view velocity")
        
        return explanations

# Example usage
async def train_niche_models():
    """
    Main function to train niche models
    """
    from supabase import create_client
    import os
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not all([supabase_url, supabase_key]):
        logger.error("[NICHE_MODELS] Missing Supabase credentials")
        return
    
    # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
    supabase = create_client(supabase_url, supabase_key)
    trainer = NicheModelTrainer(supabase)
    
    # Train models for all niches
    performance_results = await trainer.train_niche_models()
    
    logger.info(f"[NICHE_MODELS] Training completed for {len(performance_results)} niches")
    for niche, performance in performance_results.items():
        logger.info(f"[NICHE_MODELS] {niche}: R²={performance.r2_score:.3f}, Samples={performance.training_samples}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(train_niche_models())
