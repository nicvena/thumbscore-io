#!/usr/bin/env python3
"""
Simplified API that returns consistent mock results without complex processing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import random

app = FastAPI(title="Thumbscore.io Simple API", description="Simplified thumbnail scoring service")

class ThumbnailRequest(BaseModel):
    thumbnails: List[Dict[str, Any]]

class SubScores(BaseModel):
    clarity: int
    subjectProminence: int
    color_pop: int
    emotion: int
    hierarchy: int
    similarity: int
    power_words: int

class ThumbnailScore(BaseModel):
    thumbnailId: str
    ctr_score: int
    subscores: SubScores
    power_word_analysis: Dict[str, Any]
    insights: List[str]
    overlays: Dict[str, Any]

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Simple API running"}

@app.post("/v1/score")
async def score_thumbnails(request: ThumbnailRequest):
    """
    Simplified scoring that returns consistent results
    """
    
    results = []
    
    for i, thumb in enumerate(request.thumbnails):
        # Generate consistent but realistic scores
        base_score = 40 + (i * 10)  # 40, 50, 60 for 3 thumbnails
        
        # Make scores more realistic and differentiated
        if i == 0:  # Winner
            ctr_score = 65
            clarity = 45
            subject_prominence = 60
            color_pop = 50
            emotion = 55
            hierarchy = 70
            similarity = 75
            power_words = 30
        elif i == 1:  # Second place
            ctr_score = 55
            clarity = 35
            subject_prominence = 40
            color_pop = 45
            emotion = 40
            hierarchy = 60
            similarity = 70
            power_words = 30
        else:  # Third place
            ctr_score = 35
            clarity = 25
            subject_prominence = 20
            color_pop = 30
            emotion = 25
            hierarchy = 40
            similarity = 65
            power_words = 30
        
        # Create consistent power word analysis
        power_word_analysis = {
            "score": power_words,
            "found_words": [],
            "recommendation": "❌ Add bold text with 2-3 power words. Example: \"SHOCKING Results REVEALED\"",
            "warnings": ["No text detected on thumbnail"],
            "missing_opportunities": [
                "Add text overlay to your thumbnail",
                "Use 2-3 power words: INSANE, SECRET, EXPOSED",
                "Include numbers or comparisons for extra impact"
            ]
        }
        
        # Create insights based on scores
        insights = []
        if clarity < 40:
            insights.append(f"Text clarity is low ({clarity}/100) - reduce words to 3 or fewer")
        if subject_prominence < 50:
            insights.append(f"Subject size is small ({subject_prominence}/100) - make face/subject larger")
        if color_pop < 40:
            insights.append(f"Color contrast needs improvement ({color_pop}/100) - use brighter colors")
        if emotion < 40:
            insights.append(f"Emotional impact is weak ({emotion}/100) - add more expression")
        
        result = ThumbnailScore(
            thumbnailId=thumb.get("id", f"thumb_{i+1}"),
            ctr_score=ctr_score,
            subscores=SubScores(
                clarity=clarity,
                subjectProminence=subject_prominence,
                color_pop=color_pop,
                emotion=emotion,
                hierarchy=hierarchy,
                similarity=similarity,
                power_words=power_words
            ),
            power_word_analysis=power_word_analysis,
            insights=insights,
            overlays={
                "heatmap": [],
                "ocr": [],
                "faces": [],
                "grid": []
            }
        )
        
        results.append(result)
    
    # Sort by score to determine ranking
    results.sort(key=lambda x: x.ctr_score, reverse=True)
    
    # Assign rankings
    for i, result in enumerate(results):
        result.thumbnailId = f"thumb_{i+1}"
    
    return {
        "analyses": results,
        "summary": {
            "total_thumbnails": len(results),
            "winner_id": results[0].thumbnailId if results else None,
            "recommendation": f"Thumbnail {results[0].thumbnailId} scored highest with {results[0].ctr_score}/100"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
