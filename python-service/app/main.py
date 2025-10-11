"""
FastAPI Inference Service for Thumbnail Scoring
High-performance Python service that plugs into your existing Next.js app
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
from datetime import datetime

app = FastAPI(
    title="Thumbnail Scoring API",
    description="ML-powered thumbnail scoring service",
    version="1.0.0"
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
        # Fallback: return random embedding
        return np.random.randn(768).astype(np.float32)
    
    try:
        with torch.no_grad():
            image_tensor = pipeline.clip_preprocess(image).unsqueeze(0).to(pipeline.device)
            embedding = pipeline.clip_model.encode_image(image_tensor)
            return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"[CLIP] Error: {e}")
        return np.random.randn(768).astype(np.float32)

def extract_ocr_features(image: Image.Image) -> Dict[str, Any]:
    """Extract OCR features using PaddleOCR"""
    if pipeline.ocr_model is None:
        return {
            "text": "",
            "word_count": 0,
            "text_area_percent": 0,
            "contrast": 0,
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
    if pipeline.face_model is None or pipeline.emotion_model is None:
        return {
            "face_count": 0,
            "dominant_face_size": 0,
            "emotions": {"smile": 0, "anger": 0, "surprise": 0, "neutral": 0},
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
    features = {
        "clip_embedding": extract_clip_embedding(image),
        "ocr": extract_ocr_features(image),
        "faces": extract_face_features(image),
        "colors": extract_color_features(image),
        "title": title,
        "image_size": {"width": image.width, "height": image.height}
    }
    
    return features

# ============================================================================
# MODEL PREDICTION
# ============================================================================

def model_predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run ML model prediction
    Returns: CTR score and sub-scores
    """
    # If you have a trained model, use it here:
    # with torch.no_grad():
    #     embedding = torch.from_numpy(features['clip_embedding']).to(pipeline.device)
    #     prediction = pipeline.ranking_model(embedding)
    
    # For now, use heuristic-based scoring
    ocr = features['ocr']
    faces = features['faces']
    colors = features['colors']
    
    # Calculate sub-scores based on features
    clarity_score = min(100, max(0, 100 - (ocr['word_count'] * 10)))
    prominence_score = min(100, faces['dominant_face_size'] * 2.5)
    contrast_score = min(100, (colors['contrast'] / 128) * 100)
    emotion_score = min(100, faces['emotions'].get('happy', 0) * 100 + faces['emotions'].get('surprise', 0) * 100)
    hierarchy_score = 75  # Placeholder
    title_match_score = 70  # Placeholder - would use semantic similarity
    
    # Overall CTR score (weighted combination)
    ctr_score = (
        clarity_score * 0.20 +
        prominence_score * 0.25 +
        contrast_score * 0.20 +
        emotion_score * 0.15 +
        hierarchy_score * 0.10 +
        title_match_score * 0.10
    )
    
    return {
        "ctr_score": ctr_score,
        "subscores": {
            "clarity": int(clarity_score),
            "subject_prominence": int(prominence_score),
            "contrast_pop": int(contrast_score),
            "emotion": int(emotion_score),
            "hierarchy": int(hierarchy_score),
            "title_match": int(title_match_score)
        },
        "features": features
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
        "contrast_pop": "color contrast",
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
    """Initialize models on startup"""
    pipeline.initialize()

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "Thumbnail Scoring API",
        "version": "1.0.0",
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
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """
    Main inference endpoint - scores and ranks thumbnails
    """
    try:
        start_time = datetime.now()
        
        print(f"[Inference] Processing {len(req.thumbnails)} thumbnails for: '{req.title}'")
        
        # Process each thumbnail
        results = []
        for thumb in req.thumbnails:
            print(f"[Inference] Analyzing thumbnail {thumb.id}...")
            
            # 1. Extract features
            features = extract_features(thumb.url, req.title)
            
            # 2. Run model prediction
            prediction = model_predict(features)
            
            # 3. Format with explanations
            result = pred_with_explanations(thumb.id, features, prediction)
            results.append(result)
        
        # 4. Choose winner
        winner = max(results, key=lambda r: r.ctr_score)
        winner_id = winner.id
        
        # 5. Generate explanation
        explanation = explain(results, winner_id)
        
        # Sort by score (descending)
        results.sort(key=lambda r: r.ctr_score, reverse=True)
        
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

