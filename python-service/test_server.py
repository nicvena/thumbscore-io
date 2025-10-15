#!/usr/bin/env python3
"""
Minimal test server to debug the image processing issue
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import io
from PIL import Image
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Thumbscore.io Test API", version="1.0.0")

class Thumb(BaseModel):
    id: str
    url: str

class ScoreRequest(BaseModel):
    title: str
    thumbnails: List[Thumb]
    category: Optional[str] = None

class SubScores(BaseModel):
    similarity: int
    clarity: int
    subject_prominence: int
    contrast_pop: int
    emotion: int
    hierarchy: int
    title_match: int

class ThumbnailScore(BaseModel):
    id: str
    ctr_score: int
    subscores: SubScores
    confidence: float

class ScoreResponse(BaseModel):
    results: List[ThumbnailScore]
    explanation: str
    score_version: str

def load_image_from_url(url: str) -> Image.Image:
    """Download and load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from {url}: {str(e)}")

def extract_ocr_features(image: Image.Image) -> dict:
    """Simple OCR feature extraction"""
    # Simple fallback - assume some text
    return {
        "text": "Sample text",
        "word_count": 3,
        "text_area_percent": 20,
        "contrast": 85,
        "boxes": []
    }

def extract_face_features(image: Image.Image) -> dict:
    """Simple face feature extraction"""
    # Simple fallback - assume a face
    return {
        "face_count": 1,
        "dominant_face_size": 25,  # Fixed as percentage
        "emotions": {
            "happy": 0.8,
            "smile": 0.8,
            "surprise": 0.2,
            "angry": 0.05,
            "sad": 0.05,
            "fear": 0.02,
            "disgust": 0.02,
            "neutral": 0.2
        },
        "face_boxes": []
    }

def extract_color_features(image: Image.Image) -> dict:
    """Simple color feature extraction"""
    img_array = np.array(image)
    return {
        "brightness": float(np.mean(img_array)),
        "contrast": float(np.std(img_array)),
        "saturation": 50.0,
        "red_dominance": float(np.mean(img_array[:,:,0])),
        "yellow_dominance": float(np.mean(img_array[:,:,1]))
    }

def calculate_scores(ocr_features, face_features, color_features) -> dict:
    """Calculate visual subscores"""
    
    # Clarity score
    word_count = ocr_features['word_count']
    if word_count <= 3:
        clarity_score = 95
    elif word_count <= 5:
        clarity_score = 85
    elif word_count <= 8:
        clarity_score = 75
    else:
        clarity_score = 50
    
    # Prominence score
    face_size = face_features['dominant_face_size']
    if face_size >= 25:
        prominence_score = 95
    elif face_size >= 15:
        prominence_score = 85
    elif face_size >= 8:
        prominence_score = 75
    else:
        prominence_score = 65
    
    # Contrast score
    contrast_raw = color_features['contrast']
    if contrast_raw >= 50:
        contrast_score = 95
    elif contrast_raw >= 30:
        contrast_score = 85
    elif contrast_raw >= 20:
        contrast_score = 75
    else:
        contrast_score = 65
    
    # Emotion score
    emotions = face_features['emotions']
    max_emotion = max(emotions.values())
    if max_emotion >= 0.8:
        emotion_score = 95
    elif max_emotion >= 0.6:
        emotion_score = 85
    elif max_emotion >= 0.4:
        emotion_score = 75
    else:
        emotion_score = 65
    
    # Hierarchy score (simple)
    hierarchy_score = 80
    
    return {
        "clarity": clarity_score,
        "subject_prominence": prominence_score,
        "contrast_pop": contrast_score,
        "emotion": emotion_score,
        "hierarchy": hierarchy_score
    }

@app.get("/")
async def root():
    return {"service": "Thumbscore.io Test API", "status": "operational"}

@app.post("/v1/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """Test scoring endpoint"""
    try:
        logger.info(f"Processing {len(req.thumbnails)} thumbnails for: '{req.title}'")
        
        results = []
        for thumb in req.thumbnails:
            logger.info(f"Analyzing thumbnail {thumb.id}...")
            
            # Load image
            image = load_image_from_url(thumb.url)
            logger.info(f"Image loaded: {image.size}")
            
            # Extract features
            ocr_features = extract_ocr_features(image)
            face_features = extract_face_features(image)
            color_features = extract_color_features(image)
            
            logger.info(f"Features extracted - OCR: {ocr_features['word_count']} words, Face: {face_features['dominant_face_size']}%")
            
            # Calculate scores
            scores = calculate_scores(ocr_features, face_features, color_features)
            
            # Calculate overall CTR score
            ctr_score = int(sum(scores.values()) / len(scores))
            
            result = ThumbnailScore(
                id=thumb.id,
                ctr_score=ctr_score,
                subscores=SubScores(
                    similarity=75,  # Mock similarity
                    clarity=scores["clarity"],
                    subject_prominence=scores["subject_prominence"],
                    contrast_pop=scores["contrast_pop"],
                    emotion=scores["emotion"],
                    hierarchy=scores["hierarchy"],
                    title_match=80
                ),
                confidence=85.0
            )
            results.append(result)
        
        return ScoreResponse(
            results=results,
            explanation=f"Processed {len(results)} thumbnails successfully",
            score_version="test-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
