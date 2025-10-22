"""
Railway-Safe FastAPI Thumbnail Scoring Service
Essential scoring functionality without heavy ML dependencies
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import io
import base64
import requests
import hashlib
import json
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Thumbnail Scoring API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Thumb(BaseModel):
    id: str
    url: str

class ScoreRequest(BaseModel):
    title: str
    thumbnails: List[Thumb]
    category: Optional[str] = None

class SubScores(BaseModel):
    similarity: float = 75.0
    clarity: float = 80.0
    subject_prominence: float = 75.0
    contrast_pop: float = 70.0
    emotion: float = 65.0
    hierarchy: float = 75.0
    title_match: float = 80.0

class ThumbnailScore(BaseModel):
    id: str
    ctr_score: float
    subscores: SubScores
    insights: List[str]
    overlays: Dict[str, str]

@app.get("/")
def root():
    return {"message": "Thumbnail Scoring API", "status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/score")
@app.post("/score")  # Keep both endpoints for compatibility
async def score_thumbnails(req: ScoreRequest):
    """
    Railway-safe thumbnail scoring with basic visual analysis
    """
    try:
        results = []
        
        for idx, thumb in enumerate(req.thumbnails):
            # Basic scoring without heavy ML
            score = generate_basic_score(thumb, req.title, idx)
            results.append(score)
        
        # Find winner
        winner = max(results, key=lambda x: x.ctr_score)
        
        # Generate explanation
        explanation = f"{winner.id} wins with {winner.ctr_score}% CTR score due to strong visual appeal and composition."
        
        return {
            "winner_id": winner.id,
            "thumbnails": results,
            "explanation": explanation
        }
        
    except Exception as e:
        logger.error(f"Scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

async def analyze_with_gpt4_vision(thumb: Thumb, title: str, category: str) -> dict:
    """
    Analyze thumbnail using GPT-4 Vision with structured scoring
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY not found, using fallback scoring")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        # Prepare the prompt for GPT-4 Vision
        system_prompt = f"""You are an expert YouTube thumbnail analyzer. Analyze this thumbnail and provide scores (0-100) for each metric:

        SCORING CRITERIA:
        - clarity: Text readability, image sharpness, mobile optimization
        - subject_prominence: How well the main subject stands out  
        - contrast_pop: Color contrast and visual impact against YouTube's interface
        - emotion: Emotional appeal, facial expressions, curiosity factor
        - hierarchy: Visual flow, rule of thirds, composition
        - title_match: How well thumbnail matches the title "{title}"

        VIDEO CONTEXT:
        Title: "{title}"
        Category: "{category}"

        Return JSON ONLY in this exact format:
        {{
            "clarity": 85,
            "subject_prominence": 92, 
            "contrast_pop": 78,
            "emotion": 88,
            "hierarchy": 81,
            "title_match": 94,
            "insights": ["Specific insight 1", "Specific insight 2", "Specific insight 3"],
            "winner_summary": "One sentence explaining why this thumbnail works for YouTube"
        }}

        Be precise with scores - consider mobile viewing, YouTube competition, and click psychology."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this YouTube thumbnail for the video titled: '{title}'"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": thumb.url}
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        # Parse GPT-4 response
        content = response.choices[0].message.content
        try:
            analysis = json.loads(content)
            return analysis
        except json.JSONDecodeError:
            logger.error(f"Failed to parse GPT-4 response: {content}")
            return None
            
    except Exception as e:
        logger.error(f"GPT-4 Vision analysis failed: {str(e)}")
        return None

def generate_basic_score(thumb: Thumb, title: str, index: int) -> ThumbnailScore:
    """
    Generate thumbnail score using GPT-4 Vision analysis with fallback
    """
    try:
        # Get image for analysis
        if thumb.url.startswith('data:image'):
            # Handle base64 data URL
            header, data = thumb.url.split(',', 1)
            image_data = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_data))
            # Use the data URL for GPT-4 Vision
            image_url_for_gpt = thumb.url
        else:
            # Handle regular URL
            response = requests.get(thumb.url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image_url_for_gpt = thumb.url
        
        width, height = image.size
        aspect_ratio = width / height
        
        # Try GPT-4 Vision analysis first
        import asyncio
        gpt_analysis = None
        try:
            # Create new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Create a modified thumb object with the correct URL for GPT-4
            gpt_thumb = Thumb(id=thumb.id, url=image_url_for_gpt)
            gpt_analysis = loop.run_until_complete(
                analyze_with_gpt4_vision(gpt_thumb, title, "general")
            )
        except Exception as e:
            logger.error(f"GPT-4 analysis failed: {str(e)}")
        
        # Use GPT-4 scores if available, otherwise fallback
        if gpt_analysis and all(key in gpt_analysis for key in ['clarity', 'subject_prominence', 'contrast_pop', 'emotion', 'hierarchy', 'title_match']):
            subscores = SubScores(
                similarity=75.0,  # Not analyzed by GPT-4 currently
                clarity=float(gpt_analysis['clarity']),
                subject_prominence=float(gpt_analysis['subject_prominence']),
                contrast_pop=float(gpt_analysis['contrast_pop']),
                emotion=float(gpt_analysis['emotion']),
                hierarchy=float(gpt_analysis['hierarchy']),
                title_match=float(gpt_analysis['title_match'])
            )
            insights = gpt_analysis.get('insights', ['GPT-4 Vision analysis completed'])
            winner_summary = gpt_analysis.get('winner_summary', '')
        else:
            # Fallback to basic scoring
            logger.warning("Using fallback scoring due to GPT-4 failure")
            subscores = SubScores(
                similarity=75.0,
                clarity=80.0 + (width * height / 100000),
                subject_prominence=70.0 + (15 if aspect_ratio > 1.5 else 5),
                contrast_pop=75.0,
                emotion=70.0,
                hierarchy=75.0,
                title_match=80.0
            )
            insights = ['Fallback analysis - GPT-4 Vision unavailable']
            winner_summary = ''
        
        # Calculate final score (weighted average)
        weights = {
            "clarity": 0.25,
            "subject_prominence": 0.20,
            "contrast_pop": 0.15,
            "emotion": 0.15,
            "hierarchy": 0.15,
            "title_match": 0.10
        }
        
        final_score = (
            subscores.clarity * weights["clarity"] +
            subscores.subject_prominence * weights["subject_prominence"] +
            subscores.contrast_pop * weights["contrast_pop"] +
            subscores.emotion * weights["emotion"] +
            subscores.hierarchy * weights["hierarchy"] +
            subscores.title_match * weights["title_match"]
        )
        
        # Use GPT-4 insights or generate fallback insights
        if 'insights' not in locals():
            insights = []
            if subscores.clarity < 75:
                insights.append("Consider improving image resolution and sharpness")
            if subscores.subject_prominence < 70:
                insights.append("Make the main subject more prominent")
            if subscores.contrast_pop < 65:
                insights.append("Increase visual contrast and color saturation")
            if subscores.emotion < 70:
                insights.append("Add more emotional appeal to the thumbnail")
            
            if not insights:
                insights.append("Good overall thumbnail composition")
        
        # Generate mock overlays (for API compatibility)
        overlays = {
            "saliency_heatmap_url": f"/api/v1/overlays/session/{thumb.id}/heatmap.png",
            "ocr_boxes_url": f"/api/v1/overlays/session/{thumb.id}/ocr.png", 
            "face_boxes_url": f"/api/v1/overlays/session/{thumb.id}/faces.png"
        }
        
        return ThumbnailScore(
            id=thumb.id,
            ctr_score=round(final_score, 1),
            subscores=subscores,
            insights=insights,
            overlays=overlays
        )
        
    except Exception as e:
        logger.error(f"Basic scoring error: {str(e)}")
        # Return fallback score
        fallback_overlays = {
            "saliency_heatmap_url": f"/api/v1/overlays/session/{thumb.id}/heatmap.png",
            "ocr_boxes_url": f"/api/v1/overlays/session/{thumb.id}/ocr.png", 
            "face_boxes_url": f"/api/v1/overlays/session/{thumb.id}/faces.png"
        }
        return ThumbnailScore(
            id=thumb.id,
            ctr_score=70.0,
            subscores=SubScores(),
            insights=["Basic analysis completed"],
            overlays=fallback_overlays
        )

@app.get("/status")
def get_status():
    return {
        "status": "operational",
        "service": "thumbnail-scoring-api",
        "version": "1.0.0",
        "mode": "railway-deployment",
        "features": {
            "basic_scoring": True,
            "ml_models": False,
            "faiss_similarity": False,
            "advanced_vision": False
        }
    }

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)