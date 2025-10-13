"""
FastAPI Inference Service for Thumbnail Scoring
High-performance Python service that plugs into your existing Next.js app

Features:
- ML-powered thumbnail scoring and ranking
- Automated YouTube thumbnail library collection
- CLIP embeddings and similarity search
- Background job scheduling with APScheduler
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from PIL import Image
import io
import requests
import hashlib
import json
import logging
import os
from pathlib import Path
from datetime import datetime
import logging

# Import APScheduler for background jobs
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Import the thumbnail collection task
from app.tasks.collect_thumbnails import update_reference_library_sync
from app.indices import rebuild_indices_sync
from app.tasks.build_faiss_index import build_faiss_indices, get_faiss_index_info
from app.ref_library import clear_index_cache
from app.faiss_cache import load_indices, refresh_indices, get_cache_stats, is_cache_ready
from app.power_words import score_power_words

# Import YouTube Intelligence Brain
from youtube_brain.brain import YouTubeBrain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Thumbscore.io API",
    description="AI thumbnail scoring service - Visual quality, power words, and similarity intelligence",
    version="2.0.0"
)

# Enable CORS for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class Thumb(BaseModel):
    id: str
    url: str

class ScoreRequest(BaseModel):
    title: str
    thumbnails: List[Thumb]
    category: Optional[str] = None

class SubScores(BaseModel):
    similarity: int  # FAISS similarity score
    power_words: Optional[int] = None  # NEW! Power word language score
    brain_weighted: Optional[int] = None  # NEW! YouTube Intelligence Brain score
    clarity: int
    subject_prominence: int
    contrast_pop: int
    emotion: int
    hierarchy: int
    title_match: int

class Overlays(BaseModel):
    saliency_heatmap_url: str
    ocr_boxes_url: str
    face_boxes_url: str

class ThumbnailScore(BaseModel):
    id: str
    ctr_score: float
    subscores: SubScores
    insights: List[str]
    overlays: Overlays

class ScoreResponse(BaseModel):
    winner_id: str
    thumbnails: List[ThumbnailScore]
    explanation: str
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# MODEL INITIALIZATION (Global Singleton)
# ============================================================================

class ModelPipeline:
    """
    Singleton model pipeline that loads all required models once
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelPipeline] Initializing on device: {self.device}")
        
        # Load models (lazy loading on first request)
        self.clip_model = None
        self.ocr_model = None
        self.face_model = None
        self.emotion_model = None
        self.saliency_model = None
        self.ranking_model = None
        
        self.initialized = False
    
    def initialize(self):
        """Load all ML models"""
        if self.initialized:
            return
        
        print("[ModelPipeline] Loading models...")
        
        # 1. CLIP ViT-L/14 for image embeddings
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            print("[ModelPipeline] ✓ CLIP ViT-L/14 loaded")
        except Exception as e:
            print(f"[ModelPipeline] ⚠ CLIP loading failed: {e}")
            self.clip_model = None
        
        # 2. OCR Model (PaddleOCR or Tesseract)
        try:
            from paddleocr import PaddleOCR
            self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print("[ModelPipeline] ✓ PaddleOCR loaded")
        except Exception as e:
            print(f"[ModelPipeline] ⚠ PaddleOCR loading failed: {e}")
            self.ocr_model = None
        
        # 3. Face Detection (RetinaFace or MediaPipe)
        try:
            from retinaface import RetinaFace
            self.face_model = RetinaFace
            print("[ModelPipeline] ✓ RetinaFace loaded")
        except Exception as e:
            print(f"[ModelPipeline] ⚠ RetinaFace loading failed: {e}")
            self.face_model = None
        
        # 4. Emotion Recognition (FER)
        try:
            from fer import FER
            self.emotion_model = FER(mtcnn=True)
            print("[ModelPipeline] ✓ FER emotion model loaded")
        except Exception as e:
            print(f"[ModelPipeline] ⚠ FER loading failed: {e}")
            self.emotion_model = None
        
        # 5. Your trained ranking model
        try:
            # Load your trained PyTorch model
            # self.ranking_model = torch.load("models/ranking_model.pt", map_location=self.device)
            # self.ranking_model.eval()
            print("[ModelPipeline] ⚠ Custom ranking model not loaded (add your model path)")
            self.ranking_model = None
        except Exception as e:
            print(f"[ModelPipeline] ⚠ Ranking model loading failed: {e}")
            self.ranking_model = None
        
        self.initialized = True
        print("[ModelPipeline] All models initialized")

# Global model instance
pipeline = ModelPipeline()

# Global YouTube Brain instance
youtube_brain = None

# Initialize YouTube Brain
async def initialize_youtube_brain():
    """Initialize the YouTube Intelligence Brain"""
    global youtube_brain
    try:
        from supabase import create_client
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        youtube_key = os.getenv("YOUTUBE_API_KEY")
        
        if all([supabase_url, supabase_key, youtube_key]):
            supabase = create_client(supabase_url, supabase_key)
            youtube_brain = YouTubeBrain(supabase, youtube_key)
            logger.info("[BRAIN] YouTube Intelligence Brain initialized")
        else:
            logger.warning("[BRAIN] Missing environment variables for YouTube Brain")
            
    except Exception as e:
        logger.error(f"[BRAIN] Failed to initialize YouTube Brain: {e}")
        youtube_brain = None

# Initialize scheduler for background jobs
scheduler = AsyncIOScheduler()

# ============================================================================
# SCHEDULED TASK WRAPPERS
# ============================================================================

def scheduled_library_refresh_and_index_rebuild():
    """
    Scheduled task that refreshes library and rebuilds FAISS indices.
    Combines both operations for the nightly job.
    """
    logger.info("Starting scheduled library refresh and index rebuild...")
    
    try:
        # Step 1: Update reference library
        logger.info("[Thumbscore] Updating reference thumbnail library...")
        stats = update_reference_library_sync()
        logger.info(f"Library refresh completed: {stats}")
        
        # Step 2: Rebuild FAISS indices
        logger.info("Rebuilding FAISS indices...")
        index_results = build_faiss_indices()
        logger.info(f"FAISS index building completed: {index_results}")
        
        # Step 3: Refresh FAISS cache with new indices
        refresh_indices()
        logger.info("Done building FAISS indices.")
        
        # Step 4: Refresh YouTube Intelligence Brain
        logger.info("Refreshing YouTube Intelligence Brain...")
        try:
            import asyncio
            if youtube_brain:
                # Run brain refresh in async context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    brain_status = loop.run_until_complete(youtube_brain.refresh_brain())
                    logger.info(f"Brain refresh completed: {brain_status.total_patterns} patterns, {brain_status.total_trends} trends")
                finally:
                    loop.close()
            else:
                logger.warning("YouTube Brain not initialized - skipping brain refresh")
        except Exception as e:
            logger.error(f"Brain refresh failed: {e}")
        
        successful = sum(index_results.values())
        total = len(index_results)
        logger.info(f"Scheduled task completed: {successful}/{total} indices built")
        
    except Exception as e:
        logger.error(f"Scheduled task failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def load_image_from_url(url: str) -> Image.Image:
    """Download and load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from {url}: {str(e)}")

def extract_clip_embedding(image: Image.Image) -> np.ndarray:
    """Extract CLIP image embedding"""
    if pipeline.clip_model is None:
        # Fallback: return deterministic embedding based on image hash for consistency
        import hashlib
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        np.random.seed(int(img_hash[:8], 16))  # Use first 8 hex chars as seed
        embedding = np.random.randn(768).astype(np.float32)
        np.random.seed(None)  # Reset seed
        return embedding
    
    try:
        with torch.no_grad():
            image_tensor = pipeline.clip_preprocess(image).unsqueeze(0).to(pipeline.device)
            embedding = pipeline.clip_model.encode_image(image_tensor)
            return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"[CLIP] Error: {e}")
        # Fallback: return deterministic embedding for consistency
        import hashlib
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        np.random.seed(int(img_hash[:8], 16))
        embedding = np.random.randn(768).astype(np.float32)
        np.random.seed(None)
        return embedding

def extract_ocr_features(image: Image.Image) -> Dict[str, Any]:
    """Extract OCR features using PaddleOCR"""
    logger.info(f"[ocr_debug] OCR model available: {pipeline.ocr_model is not None}")
    
    if pipeline.ocr_model is None:
        logger.warning("[ocr_debug] OCR model is None - using fallback text detection")
        # Fallback estimation when model detection fails
        try:
            img_array = np.array(image)
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            
            # Simple text detection: look for high-contrast rectangular areas
            edges = np.abs(np.diff(gray, axis=1))
            text_like_areas = np.sum(edges > 50) / edges.size
            
            # Conservative estimate - assume moderate text presence when unsure
            estimated_words = min(6, max(2, int(text_like_areas * 20) + 1))  # Minimum 2 words assumed
            
            return {
                "text": f"Estimated {estimated_words} words" if estimated_words > 0 else "Moderate text",
                "word_count": estimated_words,
                "text_area_percent": min(30, max(15, text_like_areas * 100)),  # Conservative range 15-30%
                "contrast": 70,  # Conservative estimate, not perfect
                "boxes": []
            }
        except Exception as e:
            logger.error(f"[ocr_debug] Fallback failed: {e}")
            # Fallback estimation when model detection fails
            return {
                "text": "Moderate text",
                "word_count": 3,  # Conservative estimate
                "text_area_percent": 20,  # Conservative estimate
                "contrast": 60,  # Conservative estimate 
                "boxes": []
            }
    
    try:
        # Convert PIL to numpy
        img_array = np.array(image)
        result = pipeline.ocr_model.ocr(img_array, cls=True)
        
        if not result or not result[0]:
            return {"text": "", "word_count": 0, "text_area_percent": 0, "contrast": 0, "boxes": []}
        
        texts = []
        boxes = []
        for line in result[0]:
            box, (text, confidence) = line
            texts.append(text)
            boxes.append({
                "bbox": box,
                "text": text,
                "confidence": float(confidence)
            })
        
        full_text = " ".join(texts)
        
        return {
            "text": full_text,
            "word_count": len(full_text.split()),
            "text_area_percent": min(100, len(texts) * 5),  # Rough estimate
            "contrast": 75 if texts else 0,
            "boxes": boxes
        }
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return {"text": "", "word_count": 0, "text_area_percent": 0, "contrast": 0, "boxes": []}

def extract_face_features(image: Image.Image) -> Dict[str, Any]:
    """Extract face and emotion features"""
    logger.info(f"[face_debug] Face model available: {pipeline.face_model is not None}")
    logger.info(f"[face_debug] Emotion model available: {pipeline.emotion_model is not None}")
    
    if pipeline.face_model is None or pipeline.emotion_model is None:
        logger.warning("[face_debug] Face or emotion model is None - using fallback face detection")
        # Fallback estimation when model detection fails
        try:
            img_array = np.array(image)
            
            # Simple face detection: look for skin-tone colored regions
            # Convert to HSV for better skin detection
            if len(img_array.shape) == 3:
                hsv = np.array(image.convert('HSV'))
                # Skin tone ranges in HSV
                skin_mask = ((hsv[:,:,0] >= 0) & (hsv[:,:,0] <= 20)) & \
                           ((hsv[:,:,1] >= 20) & (hsv[:,:,1] <= 170)) & \
                           ((hsv[:,:,2] >= 35) & (hsv[:,:,2] <= 255))
                
                skin_pixels = np.sum(skin_mask)
                total_pixels = skin_mask.size
                face_area_ratio = skin_pixels / total_pixels
                
                # Conservative estimate - assume moderate face presence when unsure
                estimated_face_size = min(0.25, max(0.08, face_area_ratio * 2 + 0.05))  # Add baseline
                estimated_face_count = 1  # Conservative estimate - assume there's a face
                
                # Estimate emotions based on image characteristics with conservative baseline
                brightness = np.mean(img_array)
                estimated_happy = min(0.7, max(0.3, (brightness - 100) / 100 + 0.2))  # Add baseline
                
                return {
                    "face_count": estimated_face_count,
                    "dominant_face_size": estimated_face_size,
                    "emotions": {
                        "smile": estimated_happy,
                        "anger": max(0.1, 0.4 - estimated_happy),  # Conservative baseline
                        "surprise": 0.35,  # Conservative moderate surprise
                        "neutral": 0.45   # Conservative neutral baseline
                    },
                    "face_boxes": []
                }
            else:
                # Fallback estimation when model detection fails
                return {
                    "face_count": 1,  # Conservative estimate
                    "dominant_face_size": 0.12,  # Conservative moderate size
                    "emotions": {"smile": 0.4, "anger": 0.1, "surprise": 0.25, "neutral": 0.45},  # Conservative baseline
                    "face_boxes": []
                }
        except Exception as e:
            logger.error(f"[face_debug] Fallback failed: {e}")
            # Fallback estimation when model detection fails
            return {
                "face_count": 1,  # Conservative estimate
                "dominant_face_size": 0.10,  # Conservative small-medium size
                "emotions": {"smile": 0.35, "anger": 0.15, "surprise": 0.25, "neutral": 0.45},  # Conservative baseline
                "face_boxes": []
            }
    
    try:
        img_array = np.array(image)
        
        # Detect faces
        faces = pipeline.face_model.detect_faces(img_array)
        
        if not faces:
            return {
                "face_count": 0,
                "dominant_face_size": 0,
                "emotions": {"smile": 0, "anger": 0, "surprise": 0, "neutral": 0},
                "face_boxes": []
            }
        
        # Get emotions
        emotions = pipeline.emotion_model.detect_emotions(img_array)
        
        # Calculate face sizes
        face_boxes = []
        max_face_size = 0
        
        for face, emotion_data in zip(faces, emotions or []):
            bbox = face['facial_area']
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            image_area = image.width * image.height
            face_size_percent = (face_area / image_area) * 100
            max_face_size = max(max_face_size, face_size_percent)
            
            emotion_scores = emotion_data.get('emotions', {}) if emotion_data else {}
            
            face_boxes.append({
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "emotion": max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral",
                "confidence": face['score'],
                "size_percent": face_size_percent
            })
        
        avg_emotions = emotions[0]['emotions'] if emotions else {"neutral": 1.0}
        
        return {
            "face_count": len(faces),
            "dominant_face_size": max_face_size,
            "emotions": avg_emotions,
            "face_boxes": face_boxes
        }
    except Exception as e:
        print(f"[Face/Emotion] Error: {e}")
        return {
            "face_count": 0,
            "dominant_face_size": 0,
            "emotions": {"smile": 0, "anger": 0, "surprise": 0, "neutral": 1.0},
            "face_boxes": []
        }

def extract_color_features(image: Image.Image) -> Dict[str, Any]:
    """Extract color and composition features"""
    try:
        img_array = np.array(image)
        
        # Brightness
        brightness = np.mean(img_array)
        
        # Contrast (std deviation)
        contrast = np.std(img_array)
        
        # Saturation
        hsv = Image.fromarray(img_array).convert('HSV')
        saturation = np.mean(np.array(hsv)[:,:,1])
        
        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "saturation": float(saturation),
            "red_dominance": float(np.mean(img_array[:,:,0])),
            "yellow_dominance": float(np.mean((img_array[:,:,0] + img_array[:,:,1]) / 2))
        }
    except Exception as e:
        print(f"[Color] Error: {e}")
        return {
            "brightness": 128,
            "contrast": 50,
            "saturation": 128,
            "red_dominance": 128,
            "yellow_dominance": 128
        }

def extract_features(thumb_url: str, title: str) -> Dict[str, Any]:
    """
    Complete feature extraction pipeline
    Returns all features needed for model prediction
    """
    # Load image
    image = load_image_from_url(thumb_url)
    
    # Extract all features
    ocr_features = extract_ocr_features(image)
    face_features = extract_face_features(image)
    color_features = extract_color_features(image)
    
    features = {
        "clip_embedding": extract_clip_embedding(image),
        "ocr": ocr_features,
        "faces": face_features,
        "colors": color_features,
        "title": title,
        "image_size": {"width": image.width, "height": image.height}
    }
    
    return features

# ============================================================================
# MODEL PREDICTION
# ============================================================================

def amplify_score(raw_score: float) -> int:
    """
    Amplify scores with more honest scaling to prevent hiding quality issues.
    
    Maps raw scores (0-100) to more honest range (30-95) with realistic spread:
    - Excellent thumbnails: 89-95 (truly exceptional)
    - Good thumbnails: 76-88 (solid performance) 
    - Average thumbnails: 61-75 (room for improvement)
    - Poor thumbnails: 46-60 (needs significant work)
    - Very poor thumbnails: 30-45 (major issues)
    
    Args:
        raw_score: Raw score from 0-100
    
    Returns:
        Amplified score from 30-95 with honest assessment
    """
    import math
    
    # Clamp input to reasonable range
    raw_score = max(0, min(100, raw_score))
    
    # HONEST SCALING: More conservative amplification that preserves quality differences
    if raw_score < 30:
        # Very poor: map 0-30 → 30-45 (honest assessment of poor quality)
        amplified = 30 + (raw_score / 30) * 15
    elif raw_score < 50:
        # Poor: map 30-50 → 46-60 (needs significant work)
        amplified = 46 + ((raw_score - 30) / 20) * 14
    elif raw_score < 70:
        # Average: map 50-70 → 61-75 (room for improvement)
        amplified = 61 + ((raw_score - 50) / 20) * 14
    elif raw_score < 85:
        # Good: map 70-85 → 76-88 (solid performance)
        amplified = 76 + ((raw_score - 70) / 15) * 12
    else:
        # Excellent: map 85-100 → 89-95 (truly exceptional)
        amplified = 89 + ((raw_score - 85) / 15) * 6
    
    # Light smoothing to prevent harsh jumps at boundaries
    center = 62  # Center point for smoothing
    steepness = 0.06  # Gentler smoothing
    sigmoid_adjustment = 1 / (1 + math.exp(-steepness * (amplified - center)))
    
    # Minimal blending to preserve honest scoring
    final_score = amplified * 0.95 + (amplified * sigmoid_adjustment) * 0.05
    
    # Final safety clamp and round to integer
    final_score = max(30, min(95, final_score))
    
    return int(round(final_score))

def get_niche_avg_score(niche: str) -> float:
    """
    Get average similarity score for a niche (for calibration).
    Returns baseline score for niche-specific normalization.
    """
    try:
        # This would ideally query historical data or cached averages
        # For now, return reasonable baselines per niche
        niche_baselines = {
            "tech": 75.0,
            "gaming": 78.0,
            "education": 72.0,
            "entertainment": 80.0,
            "people": 82.0
        }
        return niche_baselines.get(niche, 75.0)
    except Exception as e:
        logger.debug(f"Failed to get niche baseline for {niche}: {e}")
        return 75.0  # Default baseline

async def model_predict(features: Dict[str, Any], niche: str = "tech") -> Dict[str, Any]:
    """
    Hybrid ML model prediction combining similarity intelligence with visual quality
    Returns: CTR score and sub-scores with FAISS similarity integration
    """
    # Extract visual features
    ocr = features['ocr']
    faces = features['faces']
    colors = features['colors']
    clip_embedding = features['clip_embedding']
    
    # 1. Calculate visual quality sub-scores with improved fallback handling
    # Clarity: Reward fewer words, penalize too many words heavily
    word_count = ocr['word_count']
    
    if word_count <= 3:
        clarity_score = 95  # Excellent - 3 words or less
    elif word_count <= 5:
        clarity_score = 85  # Good - 4-5 words
    elif word_count <= 8:
        clarity_score = 70  # Average - 6-8 words
    elif word_count == 0:
        clarity_score = 45  # Fallback estimation when model detection fails
    else:
        clarity_score = max(20, 85 - (word_count - 8) * 8)  # Poor - too many words
    
    # Prominence: Better face size scoring with improved fallbacks
    face_size = faces['dominant_face_size']
    face_count = faces.get('face_count', 0)
    
    if face_size >= 0.25:  # 25%+ of frame
        prominence_score = 95  # Excellent
    elif face_size >= 0.15:  # 15-25%
        prominence_score = 80  # Good
    elif face_size >= 0.08:  # 8-15%
        prominence_score = 65  # Average
    elif face_size > 0 or face_count > 0:  # Small face detected
        prominence_score = 45  # Poor but present
    else:
        prominence_score = 35  # Fallback estimation when model detection fails
    
    # Contrast: More generous scoring with better fallbacks
    contrast_raw = colors['contrast']
    contrast_ratio = contrast_raw / 128
    
    # More generous contrast scoring
    if contrast_ratio >= 1.2:
        contrast_score = 95  # Excellent contrast
    elif contrast_ratio >= 0.8:
        contrast_score = 85  # Good contrast
    elif contrast_ratio >= 0.5:
        contrast_score = 75  # Average contrast
    elif contrast_ratio >= 0.3:
        contrast_score = 65  # Below average but acceptable
    else:
        contrast_score = max(40, 50 + contrast_ratio * 50)  # Poor but not terrible
    
    # Emotion: Enhanced emotion scoring with fallback
    happy_score = faces['emotions'].get('happy', faces['emotions'].get('smile', 0)) * 100
    surprise_score = faces['emotions'].get('surprise', 0) * 100
    emotion_score = min(100, max(40, happy_score + surprise_score * 0.8 + 30))  # Boost baseline
    
    # Fallback estimation when model detection fails - if all emotions are 0, use conservative estimate  
    if happy_score == 0 and surprise_score == 0 and all(v == 0 for v in faces['emotions'].values()):
        emotion_score = 40  # Fallback estimation when model detection fails
    
    # Hierarchy: More realistic scoring based on visual elements
    hierarchy_score = 75  # Default average, could be enhanced with composition analysis
    
    # 1.5. Analyze power words in OCR text (NEW!)
    ocr_text = ocr.get('text', '')
    power_word_analysis = score_power_words(ocr_text, niche)
    power_word_score = power_word_analysis['score']
    
    logger.info(f"[POWER_WORDS] Niche '{niche}': {power_word_score:.1f}/100 - {power_word_analysis['recommendation']}")
    
    # 2. Get FAISS similarity score (intelligence-based)
    similarity_score = 75.0  # Default fallback (higher baseline)
    similarity_source = "unknown"
    
    try:
        from app.ref_library import get_similarity_score, is_cache_ready
        from app.faiss_cache import get_cache_stats
        
        # Check cache status
        if is_cache_ready():
            cache_stats = get_cache_stats()
            logger.debug(f"[FAISS] Cache status: {cache_stats['total_niches']} indices loaded, {cache_stats['total_items']} items")
        else:
            logger.warning(f"[FAISS] Cache is empty - no indices loaded")
        
        # Attempt to get similarity score
        similarity_score = get_similarity_score(clip_embedding, niche)
        
        if similarity_score is not None:
            similarity_source = "FAISS"
            logger.info(f"[SIMILARITY] Niche '{niche}': {similarity_score:.1f} (from FAISS) ✅")
        else:
            # FAISS failed, use baseline
            similarity_score = get_niche_avg_score(niche)
            similarity_source = "baseline"
            logger.warning(f"[SIMILARITY] Niche '{niche}': {similarity_score:.1f} (from baseline - FAISS unavailable) ⚠️")
            
    except Exception as e:
        logger.error(f"[SIMILARITY] Error getting similarity for {niche}: {e}")
        # Use niche-specific baseline when FAISS unavailable
        similarity_score = get_niche_avg_score(niche)
        similarity_source = "baseline_fallback"
        logger.warning(f"[SIMILARITY] Niche '{niche}': {similarity_score:.1f} (from baseline - error fallback) ⚠️")
    
    # 3. Rebalanced weights for better accuracy
    weights = {
        "similarity": 0.30,    # Reduced from 35% to prevent similarity dominance
        "power_words": 0.10,   # Language quality
        "clarity": 0.22,       # Increased from 20% - text clarity is crucial
        "color_pop": 0.15,     # Visual appeal
        "emotion": 0.10,       # Emotional impact
        "hierarchy": 0.13      # Increased from 10% - visual structure important
    }
    
    # Map visual scores to weight keys
    visual_scores = {
        "similarity": similarity_score,
        "power_words": power_word_score,  # NEW!
        "clarity": clarity_score,
        "color_pop": contrast_score,
        "emotion": emotion_score,
        "hierarchy": hierarchy_score
    }
    
    # 4. Compute weighted CTR score (raw)
    raw_ctr_score = sum(weights[k] * visual_scores[k] for k in weights)
    
    # 4.5. CRITICAL QUALITY GATES SYSTEM - Apply BEFORE amplification
    quality_gates_applied = []
    
    # Gate 1: If clarity is very poor, cap score
    if clarity_score < 20:
        raw_ctr_score = min(raw_ctr_score, 55)
        quality_gates_applied.append("clarity too low")
        logger.warning(f"Quality gate applied: clarity too low ({clarity_score}), capping score at 55")
    
    # Gate 2: If subject prominence is very poor, cap score  
    if prominence_score < 20:
        raw_ctr_score = min(raw_ctr_score, 60)
        quality_gates_applied.append("subject prominence too low")
        logger.warning(f"Quality gate applied: subject prominence too low ({prominence_score}), capping score at 60")
    
    # Gate 3: If multiple visual components are critically poor, severe cap
    poor_visual_count = sum(1 for score in [clarity_score, prominence_score, hierarchy_score] if score < 10)
    if poor_visual_count >= 2:
        raw_ctr_score = min(raw_ctr_score, 50)
        quality_gates_applied.append("multiple visual components critically poor")
        logger.warning(f"Quality gate applied: {poor_visual_count} visual components < 10, capping score at 50")
    
    # Log quality gates summary
    if quality_gates_applied:
        logger.info(f"[QUALITY_GATES] Applied: {', '.join(quality_gates_applied)}")
    
    # 5. Light niche calibration (less aggressive for better scores)
    try:
        niche_mean = get_niche_avg_score(niche)
        if niche_mean > 0 and niche_mean != 75.0:  # Only calibrate if we have real data
            # More conservative calibration - just slight adjustment
            calibration_factor = min(1.1, max(0.9, 75.0 / niche_mean))  # Max 10% adjustment
            raw_ctr_score = raw_ctr_score * calibration_factor
            raw_ctr_score = min(max(raw_ctr_score, 0), 100)  # Clamp to [0, 100]
            logger.debug(f"[calibration] niche={niche} mean={niche_mean:.1f} factor={calibration_factor:.2f} → {raw_ctr_score:.1f}")
    except Exception as e:
        logger.debug(f"Niche calibration failed for {niche}: {e}")
    
    # 6. Apply score amplification for better differentiation
    final_score = amplify_score(raw_ctr_score)
    
    # 6.5. CONSISTENCY CHECK - High scores must have decent visual quality
    if final_score > 60:
        # At least one visual component (clarity, subject, hierarchy) must be > 40
        visual_components = [clarity_score, prominence_score, hierarchy_score]
        if all(score <= 40 for score in visual_components):
            # Apply 15% penalty for inconsistency
            consistency_penalty = int(final_score * 0.15)
            final_score = max(30, final_score - consistency_penalty)
            logger.warning(f"Consistency penalty applied: high score ({final_score + consistency_penalty}) but poor visuals (clarity:{clarity_score}, prominence:{prominence_score}, hierarchy:{hierarchy_score}), reduced by {consistency_penalty}")
    
    # 6.75. Add comprehensive model failure and scoring logging
    ocr_status = "success" if pipeline.ocr_model is not None else "fallback"
    face_status = "success" if (pipeline.face_model is not None and pipeline.emotion_model is not None) else "fallback"
    
    logger.info(f"[MODEL_STATUS] OCR detection: {ocr_status}, Face detection: {face_status}")
    logger.info(f"[SCORING] Raw score: {raw_ctr_score:.1f}, Amplified score: {final_score}")
    if quality_gates_applied:
        logger.info(f"[SCORING] Quality gates applied: {quality_gates_applied}")
    
    # 7. YouTube Intelligence Brain scoring (if available)
    brain_score = None
    brain_weighted_score = None
    if youtube_brain:
        try:
            # Prepare features for brain analysis
            brain_features = {
                "thumbnail_id": features.get('thumbnail_id', 'unknown'),
                "title": features.get('title', ''),
                "tags": features.get('tags', []),
                "duration": features.get('duration', 'PT0S'),
                "views_per_hour": features.get('views_per_hour', 100),
                "engagement_rate": features.get('engagement_rate', 0.03),
                "hours_ago": features.get('hours_ago', 24)
            }
            
            # Get brain score
            brain_result = await youtube_brain.score_thumbnail(
                clip_embedding, niche, brain_features
            )
            
            brain_score = brain_result.brain_weighted_score
            brain_weighted_score = int(brain_score * 100)  # Convert to 0-100 scale
            
            logger.info(f"[BRAIN] Brain score: {brain_score:.3f} (confidence: {brain_result.confidence:.3f})")
            
            # Blend brain score with final score (30% brain, 70% original)
            if brain_result.confidence > 0.7:  # Stricter confidence gate
                final_score = final_score * 0.7 + brain_weighted_score * 0.3
                logger.info(f"[BRAIN] Blended score: {final_score:.1f}")
            
        except Exception as e:
            logger.error(f"[BRAIN] Error in brain scoring: {e}")
            brain_score = None
            brain_weighted_score = None

    # NOTE: Old visual penalty system removed - replaced with quality gates system above
    
    # 8. Amplify all subscores for consistency
    amplified_subscores = {
        "similarity": amplify_score(similarity_score),
        "power_words": amplify_score(power_word_score),
        "brain_weighted": brain_weighted_score if brain_weighted_score else amplify_score(similarity_score),  # Fallback to similarity
        "clarity": amplify_score(clarity_score),
        "subject_prominence": amplify_score(prominence_score),
        "color_pop": amplify_score(contrast_score),
        "emotion": amplify_score(emotion_score),
        "hierarchy": amplify_score(hierarchy_score),
        "title_match": amplify_score(similarity_score)  # Use similarity as title match proxy
    }
    
    # 8. Enhanced logging for debugging
    logger.info(f"[SCORE] Niche '{niche}' - Raw: {raw_ctr_score:.1f} → Final: {final_score}")
    logger.info(f"[SUBS] Sim: {similarity_score:.0f}→{amplified_subscores['similarity']}, Power: {power_word_score:.0f}→{amplified_subscores['power_words']}, Clarity: {clarity_score:.0f}→{amplified_subscores['clarity']}")
    
    return {
        "ctr_score": final_score,  # Amplified final score
        "subscores": amplified_subscores,  # All amplified subscores
        "features": features,
        "weights_used": weights,
        "niche": niche,
        "similarity_source": similarity_source,  # Track where similarity came from
        "raw_score": raw_ctr_score,  # Keep for debugging (don't show to user)
        "power_word_analysis": power_word_analysis  # NEW! Full power word details
    }

def pred_with_explanations(thumb_id: str, features: Dict[str, Any], prediction: Dict[str, Any]) -> ThumbnailScore:
    """
    Convert prediction to response format with insights
    """
    subscores = prediction['subscores']
    insights = []
    
    # Generate insights based on sub-scores
    if subscores['clarity'] < 70:
        insights.append(f"Reduce words from {features['ocr']['word_count']}→3; use bold block font")
        insights.append("Boost contrast between text and background")
    
    if subscores['subject_prominence'] < 70:
        insights.append("Increase subject size ~25%")
        insights.append("Center main subject for better attention capture")
    
    if subscores['contrast_pop'] < 70:
        insights.append("Boost saturation by 15-20% for more visual pop")
    
    if subscores['emotion'] < 70:
        insights.append("Add more expressive facial emotion or action")
    
    # Generate overlay URLs
    session_id = f"py_{datetime.now().timestamp()}"
    overlays = Overlays(
        saliency_heatmap_url=f"/api/v1/overlays/{session_id}/{thumb_id}/heatmap.png",
        ocr_boxes_url=f"/api/v1/overlays/{session_id}/{thumb_id}/ocr.png",
        face_boxes_url=f"/api/v1/overlays/{session_id}/{thumb_id}/faces.png"
    )
    
    return ThumbnailScore(
        id=thumb_id,
        ctr_score=round(prediction['ctr_score'], 1),
        subscores=SubScores(**subscores),
        insights=insights[:5],  # Top 5 insights
        overlays=overlays
    )

def explain(results: List[ThumbnailScore], winner_id: str) -> str:
    """Generate explanation for why a thumbnail won"""
    winner = next(r for r in results if r.id == winner_id)
    
    # Find strongest sub-scores
    subscores = winner.subscores.dict()
    top_scores = sorted(subscores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    score_names = {
        "clarity": "text clarity",
        "subject_prominence": "face/subject prominence",
        "color_pop": "color contrast",
        "emotion": "emotional appeal",
        "hierarchy": "visual hierarchy",
        "title_match": "title alignment"
    }
    
    reasons = [score_names[k] for k, v in top_scores if v >= 75]
    
    if reasons:
        return f"{winner_id} wins due to {', '.join(reasons)}."
    else:
        return f"{winner_id} wins with the highest overall CTR score."

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models and scheduler on startup"""
    logger.info("[Thumbscore.io] Starting up AI thumbnail scoring service...")
    
    # Initialize models
    pipeline.initialize()
    
    # Initialize YouTube Intelligence Brain
    await initialize_youtube_brain()
    
    # Preload FAISS indices into memory cache
    logger.info("=" * 70)
    logger.info("[FAISS] Starting FAISS index verification...")
    logger.info("=" * 70)
    
    load_indices()
    
    if is_cache_ready():
        cache_stats = get_cache_stats()
        logger.info(f"[FAISS] ✅ Cache ready with {cache_stats['total_niches']} niches")
        logger.info(f"[FAISS] Total items: {cache_stats['total_items']}")
        logger.info(f"[FAISS] Memory usage: ~{cache_stats['estimated_memory_mb']:.1f} MB")
        
        # Log which niches have indices
        logger.info("[FAISS] Available indices:")
        for niche in cache_stats.get('niches', []):
            logger.info(f"[FAISS]   ✓ {niche}")
        
        logger.info("[FAISS] 🎯 FAISS similarity scoring ACTIVE")
    else:
        logger.warning("[FAISS] ❌ No indices loaded - cache is EMPTY")
        logger.warning("[FAISS] FAISS similarity scoring will NOT be available")
        logger.warning("[FAISS] All scores will fall back to niche baselines")
        logger.warning("[FAISS]")
        logger.warning("[FAISS] To enable FAISS similarity scoring:")
        logger.warning("[FAISS]   1. Ensure you have thumbnails in Supabase")
        logger.warning("[FAISS]   2. Run: curl http://localhost:8000/internal/rebuild-indices")
        logger.warning("[FAISS]   3. Or wait for the nightly refresh at 3 AM Hobart time")
        logger.warning("[FAISS]")
        logger.warning("[FAISS] Missing index files in: faiss_indices/")
        import os
        expected_files = ["tech.index", "gaming.index", "entertainment.index", "people.index", "education.index"]
        for filename in expected_files:
            filepath = os.path.join("faiss_indices", filename)
            if os.path.exists(filepath):
                logger.info(f"[FAISS]   ✓ Found: {filename}")
            else:
                logger.warning(f"[FAISS]   ✗ Missing: {filename}")
    
    logger.info("=" * 70)
    
    # Add scheduled job for thumbnail collection + FAISS index rebuilding
    # (daily at 3 AM Hobart time)
    # Hobart is UTC+10/+11 (AEST/AEDT)
    scheduler.add_job(
        scheduled_library_refresh_and_index_rebuild,
        trigger=CronTrigger(hour=3, minute=0, timezone="Australia/Hobart"),
        id="refresh_and_index",
        name="Collect thumbnails and rebuild FAISS indices",
        replace_existing=True,
        max_instances=1
    )
    
    # Start the scheduler
    scheduler.start()
    logger.info("Scheduler started:")
    logger.info("  - Library refresh + FAISS index rebuilding: 3:00 AM Hobart time daily")
    
    # Run initial collection and index building (optional - comment out for production)
    # logger.info("Running initial setup...")
    # try:
    #     scheduled_library_refresh_and_index_rebuild()
    # except Exception as e:
    #     logger.error(f"Initial setup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    logger.info("[Thumbscore.io] Shutting down AI thumbnail scoring service...")
    scheduler.shutdown()

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "Thumbscore.io API",
        "version": "2.0.0",
        "status": "operational",
        "device": str(pipeline.device),
        "models_loaded": pipeline.initialized
    }

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "clip": pipeline.clip_model is not None,
            "ocr": pipeline.ocr_model is not None,
            "face": pipeline.face_model is not None,
            "emotion": pipeline.emotion_model is not None,
            "ranking": pipeline.ranking_model is not None
        },
        "device": str(pipeline.device),
        "gpu_available": torch.cuda.is_available(),
        "scheduler": {
            "running": scheduler.running,
            "jobs": len(scheduler.get_jobs())
        }
    }

@app.get("/internal/refresh-library")
def refresh_library():
    """
    Manually trigger thumbnail library refresh
    Also rebuilds FAISS indices after refresh
    Internal endpoint for testing and manual updates
    """
    try:
        logger.info("Manual thumbnail library refresh requested")
        
        # Step 1: Update reference library
        logger.info("[Thumbscore] Updating reference thumbnail library...")
        stats = update_reference_library_sync()
        logger.info(f"Library refresh completed: {stats}")
        
        # Step 2: Rebuild FAISS indices
        logger.info("Rebuilding FAISS indices...")
        index_results = build_faiss_indices()
        logger.info(f"FAISS index building completed: {index_results}")
        
        # Step 3: Refresh FAISS cache with new indices
        refresh_indices()
        logger.info("Done building FAISS indices.")
        
        successful_indices = sum(index_results.values())
        total_indices = len(index_results)
        
        return {
            "status": "ok",
            "message": f"Reference library and FAISS indices refreshed successfully ({successful_indices}/{total_indices} indices)",
            "library_stats": stats,
            "index_results": index_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Manual refresh failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to refresh library: {str(e)}"
        )

@app.get("/internal/rebuild-indices")
def rebuild_indices():
    """
    Manually trigger FAISS index rebuilding
    Uses the new build_faiss_index module for consistency
    Internal endpoint for testing and manual updates
    """
    try:
        logger.info("Manual FAISS index rebuilding requested")
        
        # Rebuild FAISS indices
        logger.info("Rebuilding FAISS indices...")
        results = build_faiss_indices()
        
        # Refresh FAISS cache with new indices
        refresh_indices()
        logger.info("Done building FAISS indices.")
        
        successful = sum(results.values())
        total = len(results)
        
        return {
            "status": "ok",
            "message": f"FAISS indices rebuilt successfully ({successful}/{total} niches)",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Manual index rebuilding failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to rebuild indices: {str(e)}"
        )

@app.get("/internal/faiss-status")
def get_faiss_status():
    """
    Comprehensive FAISS status check with sample similarity scores
    Shows which niches have indices and tests similarity matching
    """
    try:
        import numpy as np
        from app.ref_library import get_similarity_score
        
        # Check if cache is ready
        cache_ready = is_cache_ready()
        cache_stats = get_cache_stats() if cache_ready else {}
        
        # Get file-based index info
        index_info = get_faiss_index_info()
        
        # Test sample similarity scores for each niche
        sample_scores = {}
        if cache_ready:
            # Generate a random test embedding
            test_embedding = np.random.randn(768).astype(np.float32)
            test_embedding = test_embedding / np.linalg.norm(test_embedding)
            
            for niche in cache_stats.get('niches', []):
                try:
                    similarity = get_similarity_score(test_embedding, niche)
                    if similarity is not None:
                        sample_scores[niche] = {
                            "similarity_score": round(float(similarity), 2),
                            "source": "FAISS",
                            "status": "✅ Working"
                        }
                    else:
                        sample_scores[niche] = {
                            "similarity_score": None,
                            "source": "Failed",
                            "status": "❌ Not working"
                        }
                except Exception as e:
                    sample_scores[niche] = {
                        "similarity_score": None,
                        "source": f"Error: {str(e)}",
                        "status": "❌ Error"
                    }
        
        # Determine overall status
        if cache_ready and len(sample_scores) > 0:
            working_count = sum(1 for s in sample_scores.values() if s['status'] == "✅ Working")
            if working_count == len(sample_scores):
                overall_status = "✅ FULLY OPERATIONAL"
            elif working_count > 0:
                overall_status = f"⚠️ PARTIALLY WORKING ({working_count}/{len(sample_scores)} niches)"
            else:
                overall_status = "❌ NOT WORKING"
        else:
            overall_status = "❌ NOT LOADED"
        
        return {
            "overall_status": overall_status,
            "cache_ready": cache_ready,
            "cache_stats": cache_stats,
            "index_files": index_info,
            "sample_similarity_scores": sample_scores,
            "instructions": {
                "if_not_loaded": [
                    "1. Ensure you have thumbnails in Supabase",
                    "2. Run: curl -X POST http://localhost:8000/internal/rebuild-indices",
                    "3. Wait for indices to build (~30-60 seconds)",
                    "4. Check this endpoint again"
                ],
                "if_partially_working": [
                    "Some niches are missing data in Supabase",
                    "Run the thumbnail collector to fetch more data",
                    "curl -X POST http://localhost:8000/internal/refresh-library"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting FAISS status: {e}")
        return {
            "overall_status": "❌ ERROR",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/internal/index-stats")
def get_index_stats_endpoint():
    """
    Get FAISS index statistics (legacy endpoint)
    Use /internal/faiss-status for more detailed information
    """
    try:
        from app.ref_library import get_index_cache_stats
        
        # Get file-based index info
        index_info = get_faiss_index_info()
        
        # Get cache stats
        cache_stats = get_cache_stats()
        
        return {
            "status": "ok",
            "index_info": index_info,
            "cache_stats": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get index stats: {str(e)}"
        )

@app.get("/internal/brain-status")
async def get_brain_status():
    """
    Get YouTube Intelligence Brain status and statistics
    """
    try:
        if not youtube_brain:
            return {
                "status": "not_initialized",
                "message": "YouTube Intelligence Brain not initialized",
                "components": {
                    "data_collector": False,
                    "pattern_miner": False,
                    "niche_models": False,
                    "trend_detector": False,
                    "insights_engine": False
                }
            }
        
        brain_status = await youtube_brain.get_status()
        
        return {
            "status": "initialized" if brain_status.niche_models_ready else "partial",
            "message": "YouTube Intelligence Brain operational" if brain_status.niche_models_ready else "Brain partially initialized",
            "components": {
                "data_collector": brain_status.data_collector_ready,
                "pattern_miner": brain_status.pattern_miner_ready,
                "niche_models": brain_status.niche_models_ready,
                "trend_detector": brain_status.trend_detector_ready,
                "insights_engine": brain_status.insights_engine_ready
            },
            "statistics": {
                "total_patterns": brain_status.total_patterns,
                "total_trends": brain_status.total_trends,
                "trained_niches": brain_status.trained_niches,
                "last_update": brain_status.last_data_update.isoformat() if brain_status.last_data_update else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting brain status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get brain status: {str(e)}"
        )

@app.get("/internal/trending-patterns/{niche}")
async def get_trending_patterns(niche: str):
    """
    Get trending patterns for a specific niche
    """
    try:
        if not youtube_brain:
            raise HTTPException(
                status_code=503,
                detail="YouTube Intelligence Brain not initialized"
            )
        
        trends = await youtube_brain.get_trending_patterns(niche)
        
        return {
            "niche": niche,
            "trends": [
                {
                    "trend_id": trend.trend_id,
                    "trend_type": trend.trend_type,
                    "trend_strength": trend.trend_strength,
                    "growth_rate": trend.growth_rate,
                    "description": trend.trend_description,
                    "confidence": trend.confidence,
                    "predicted_lifespan": trend.predicted_lifespan
                }
                for trend in trends
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting trending patterns: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trending patterns: {str(e)}"
        )

@app.post("/internal/refresh-brain")
async def refresh_brain():
    """
    Refresh the YouTube Intelligence Brain with new data
    """
    try:
        if not youtube_brain:
            raise HTTPException(
                status_code=503,
                detail="YouTube Intelligence Brain not initialized"
            )
        
        logger.info("[BRAIN] Manual brain refresh requested")
        brain_status = await youtube_brain.refresh_brain()
        
        return {
            "status": "success",
            "message": "YouTube Intelligence Brain refreshed successfully",
            "brain_status": {
                "total_patterns": brain_status.total_patterns,
                "total_trends": brain_status.total_trends,
                "trained_niches": brain_status.trained_niches
            }
        }
        
    except Exception as e:
        logger.error(f"Error refreshing brain: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh brain: {str(e)}"
        )

@app.post("/v1/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """
    Main inference endpoint - scores and ranks thumbnails
    """
    try:
        start_time = datetime.now()
        
        print(f"[Inference] Processing {len(req.thumbnails)} thumbnails for: '{req.title}'")
        
        # Determine niche from request
        niche = req.category or "tech"  # Default to tech if not specified
        
        # Process each thumbnail
        results = []
        for thumb in req.thumbnails:
            print(f"[Inference] Analyzing thumbnail {thumb.id} for niche '{niche}'...")
            
            # 1. Extract features
            features = extract_features(thumb.url, req.title)
            
            # 2. Run hybrid model prediction with niche
            prediction = await model_predict(features, niche)
            
            # 3. Format with explanations
            result = pred_with_explanations(thumb.id, features, prediction)
            results.append(result)
        
        # 4. Choose winner with visual tie-breaker when close
        results.sort(key=lambda r: r.ctr_score, reverse=True)
        winner = results[0]
        if len(results) > 1:
            second = results[1]
            if abs(winner.ctr_score - second.ctr_score) <= 3:
                def visual_sum(r):
                    s = r.subscores
                    return (
                        s.clarity + s.subject_prominence + s.contrast_pop + s.emotion + s.hierarchy
                    )
                if visual_sum(second) > visual_sum(winner):
                    winner = second
        winner_id = winner.id
        
        # 5. Generate explanation
        explanation = explain(results, winner_id)
        
        # Results are already sorted by score (and possibly adjusted by tie-break)
        
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"[Inference] Completed in {duration:.0f}ms. Winner: {winner_id} ({winner.ctr_score}%)")
        
        return ScoreResponse(
            winner_id=winner_id,
            thumbnails=results,
            explanation=explanation,
            metadata={
                "processing_time_ms": round(duration),
                "model_version": "1.0.0",
                "device": str(pipeline.device)
            }
        )
        
    except Exception as e:
        print(f"[Inference] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

