"""
Example integration of niche insights into FastAPI analyze endpoint

This shows how to properly integrate the niche-specific insights system
into your thumbnail analysis backend.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

# Import your existing modules
from niche_insights import get_niche_insights, validate_niche_id, get_niche_baseline
from power_words import analyze_power_words  # Your existing power words module
# from your_scoring_module import calculate_scores  # Your scoring logic

app = FastAPI()
logger = logging.getLogger(__name__)

class ThumbnailAnalyzeRequest(BaseModel):
    session_id: str
    thumbnails: List[Dict[str, Any]]
    title: Optional[str] = None
    niche: Optional[str] = 'general'  # Add niche parameter

class ComponentScores(BaseModel):
    emotion: float
    color_pop: float
    text_clarity: float
    composition: float
    subject_prominence: float
    similarity: float

class ThumbnailResult(BaseModel):
    id: int
    filename: str
    overall_score: float
    components: ComponentScores
    power_words: Optional[Dict[str, Any]] = None
    niche_insights: Optional[List[str]] = None
    recommendations: List[Dict[str, Any]]

@app.post("/api/analyze")
async def analyze_thumbnails(request: ThumbnailAnalyzeRequest):
    """
    Analyze thumbnails with niche-specific scoring and insights
    """
    try:
        # Validate niche
        niche_id = request.niche or 'general'
        if not validate_niche_id(niche_id):
            logger.warning(f"Invalid niche '{niche_id}', falling back to 'general'")
            niche_id = 'general'
        
        logger.info(f"Starting analysis for {len(request.thumbnails)} thumbnails in '{niche_id}' niche")
        
        results = []
        thumbnail_scores = []
        
        # Process each thumbnail
        for i, thumbnail_data in enumerate(request.thumbnails):
            thumbnail_id = i + 1
            filename = thumbnail_data.get('filename', f'thumbnail_{thumbnail_id}')
            
            logger.info(f"Processing thumbnail {thumbnail_id}: {filename}")
            
            # =================================================================
            # YOUR EXISTING SCORING LOGIC HERE
            # =================================================================
            
            # Example scoring (replace with your actual ML pipeline)
            components = {
                'emotion': calculate_emotion_score(thumbnail_data, niche_id),
                'color_pop': calculate_color_score(thumbnail_data, niche_id),
                'text_clarity': calculate_clarity_score(thumbnail_data, niche_id),
                'composition': calculate_composition_score(thumbnail_data, niche_id),
                'subject_prominence': calculate_subject_score(thumbnail_data, niche_id),
                'similarity': calculate_similarity_score(thumbnail_data, niche_id)
            }
            
            # Apply niche-specific weights
            niche_weights = get_niche_weights(niche_id)
            weighted_score = apply_niche_weights(components, niche_weights)
            
            # Calibrate with niche baseline
            baseline = get_niche_baseline(niche_id)
            final_score = calibrate_score_with_baseline(weighted_score, baseline)
            
            # =================================================================
            # NICHE INSIGHTS INTEGRATION
            # =================================================================
            
            # Get niche-specific insights based on component scores
            niche_insights = get_niche_insights(niche_id, components)
            
            # Power words analysis (if title provided)
            power_words_result = None
            if request.title:
                power_words_result = analyze_power_words(
                    text=request.title,
                    niche=niche_id
                )
            
            # Generate recommendations
            recommendations = generate_niche_recommendations(components, niche_id)
            
            # Create result object
            result = {
                'id': thumbnail_id,
                'filename': filename,
                'overall_score': round(final_score, 1),
                'components': components,
                'power_words': power_words_result,
                'niche_insights': niche_insights,
                'recommendations': recommendations
            }
            
            results.append(result)
            thumbnail_scores.append(final_score)
            
            # Log niche-specific analysis details
            logger.info(f"Thumbnail {thumbnail_id} niche analysis complete:")
            logger.info(f"  - Niche: {niche_id}")
            logger.info(f"  - Final Score: {final_score}")
            logger.info(f"  - Insights Generated: {len(niche_insights)}")
            logger.info(f"  - Power Words: {len(power_words_result.get('found_words', [])) if power_words_result else 0}")
        
        # Find winner
        winner_index = thumbnail_scores.index(max(thumbnail_scores))
        winner_score = max(thumbnail_scores)
        
        # =================================================================
        # RESPONSE WITH NICHE DATA
        # =================================================================
        
        return {
            'session_id': request.session_id,
            'overall_score': winner_score,
            'niche_used': niche_id,
            'niche_baseline': get_niche_baseline(niche_id),
            'niche_insights': results[winner_index]['niche_insights'],  # Winner's insights
            'winner': {
                'id': results[winner_index]['id'],
                'score': winner_score,
                'insights': results[winner_index]['niche_insights']
            },
            'thumbnails': results,
            'metadata': {
                'analysis_type': 'niche_optimized',
                'niche_id': niche_id,
                'title_provided': bool(request.title),
                'power_words_analyzed': bool(request.title),
                'insights_generated': True,
                'timestamp': datetime.utcnow().isoformat(),
                'version': '2.0.0'
            }
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# =================================================================
# HELPER FUNCTIONS (implement based on your existing system)
# =================================================================

def get_niche_weights(niche_id: str) -> Dict[str, float]:
    """Get niche-specific scoring weights"""
    weights = {
        'gaming': {
            'emotion': 0.25,
            'color_pop': 0.20,
            'text_clarity': 0.15,
            'composition': 0.15,
            'subject_prominence': 0.15,
            'similarity': 0.10
        },
        'business': {
            'text_clarity': 0.30,
            'composition': 0.25,
            'similarity': 0.25,
            'color_pop': 0.10,
            'subject_prominence': 0.05,
            'emotion': 0.05
        },
        'education': {
            'text_clarity': 0.30,
            'similarity': 0.25,
            'composition': 0.20,
            'color_pop': 0.15,
            'subject_prominence': 0.05,
            'emotion': 0.05
        },
        # Add other niches...
        'general': {
            'similarity': 0.20,
            'text_clarity': 0.20,
            'composition': 0.20,
            'color_pop': 0.15,
            'emotion': 0.15,
            'subject_prominence': 0.10
        }
    }
    return weights.get(niche_id, weights['general'])

def apply_niche_weights(components: Dict[str, float], weights: Dict[str, float]) -> float:
    """Apply niche-specific weights to component scores"""
    weighted_score = 0.0
    for component, score in components.items():
        weight = weights.get(component, 0.0)
        weighted_score += score * weight
    return weighted_score

def calibrate_score_with_baseline(score: float, baseline: int) -> float:
    """Calibrate score with niche-specific baseline"""
    # Adjust score relative to niche baseline
    calibrated = score + (baseline - 70) * 0.1  # 70 is general baseline
    return max(0, min(100, calibrated))

def calculate_emotion_score(thumbnail_data: Dict, niche_id: str) -> float:
    """Calculate emotion score with niche adjustments"""
    # Your existing emotion detection logic here
    base_score = 75.0  # Placeholder
    
    # Niche-specific adjustments
    if niche_id == 'gaming':
        base_score *= 1.1  # Gaming values emotion more
    elif niche_id == 'business':
        base_score *= 0.9  # Business is more conservative
    
    return min(100.0, base_score)

def calculate_color_score(thumbnail_data: Dict, niche_id: str) -> float:
    """Calculate color pop score with niche adjustments"""
    # Your existing color analysis logic here
    base_score = 70.0  # Placeholder
    
    if niche_id in ['gaming', 'food', 'entertainment']:
        base_score *= 1.15  # These niches love vibrant colors
    elif niche_id in ['business', 'tech']:
        base_score *= 0.95  # More subdued professional look
    
    return min(100.0, base_score)

def calculate_clarity_score(thumbnail_data: Dict, niche_id: str) -> float:
    """Calculate text clarity with niche focus"""
    # Your OCR and text analysis logic here
    base_score = 80.0  # Placeholder
    
    if niche_id in ['business', 'education']:
        base_score *= 1.2  # These niches heavily value clarity
    
    return min(100.0, base_score)

def calculate_composition_score(thumbnail_data: Dict, niche_id: str) -> float:
    """Calculate composition score"""
    # Your composition analysis logic here
    return 75.0  # Placeholder

def calculate_subject_score(thumbnail_data: Dict, niche_id: str) -> float:
    """Calculate subject prominence"""
    # Your subject detection logic here
    base_score = 70.0  # Placeholder
    
    if niche_id == 'fitness':
        base_score *= 1.1  # Fitness values subject prominence
    
    return min(100.0, base_score)

def calculate_similarity_score(thumbnail_data: Dict, niche_id: str) -> float:
    """Calculate similarity to successful thumbnails in niche"""
    # Your FAISS similarity search logic here
    return 85.0  # Placeholder

def generate_niche_recommendations(components: Dict[str, float], niche_id: str) -> List[Dict[str, Any]]:
    """Generate niche-specific recommendations"""
    recommendations = []
    
    # Find lowest scoring components
    sorted_components = sorted(components.items(), key=lambda x: x[1])
    
    for component, score in sorted_components[:3]:  # Top 3 issues
        if score < 70:  # Needs improvement
            rec = generate_component_recommendation(component, score, niche_id)
            recommendations.append(rec)
    
    return recommendations

def generate_component_recommendation(component: str, score: float, niche_id: str) -> Dict[str, Any]:
    """Generate specific recommendation for component"""
    # Niche-specific recommendation templates
    recommendations = {
        'gaming': {
            'emotion': 'Add more exaggerated facial expressions - gaming audiences love dramatic reactions!',
            'color_pop': 'Boost saturation to 80-100% - gaming thumbnails need to pop in feeds!',
            'text_clarity': 'Use bold, outlined text with high contrast - make it readable on mobile!'
        },
        'business': {
            'text_clarity': 'Simplify your message to 2-3 words max - business viewers scan quickly',
            'composition': 'Clean up the layout - remove distracting elements for professional look'
        },
        # Add more niche-specific recommendations...
    }
    
    niche_recs = recommendations.get(niche_id, {})
    suggestion = niche_recs.get(component, f'Improve your {component.replace("_", " ")} score')
    
    priority = 'high' if score < 50 else 'medium' if score < 70 else 'low'
    
    return {
        'component': component,
        'current_score': score,
        'priority': priority,
        'suggestion': suggestion,
        'niche_specific': True,
        'expected_improvement': f'+{15-25 if priority == "high" else 10-15}% CTR'
    }

# =================================================================
# EXAMPLE USAGE
# =================================================================

if __name__ == "__main__":
    # Test the integration
    sample_request = ThumbnailAnalyzeRequest(
        session_id="test_123",
        thumbnails=[
            {"filename": "thumb1.jpg", "data": "..."},
            {"filename": "thumb2.jpg", "data": "..."}
        ],
        title="INSANE Gaming Strategy That BROKE The Game!",
        niche="gaming"
    )
    
    # This would be called by your FastAPI server
    # result = await analyze_thumbnails(sample_request)
    print("Niche backend integration ready!")