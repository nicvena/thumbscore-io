#!/usr/bin/env python3
"""
Stable V1 Thumbnail Scoring System

Deterministic results using GPT-4 Vision rubric + numeric core
- 55% GPT-4 rubric (strict JSON schema)
- 45% deterministic numeric core
- Full caching with hash-based deduplication
- Score range: 30-95
"""

import os
import base64
import io
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import pytesseract
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import statistics
from functools import lru_cache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import GPT summary module
try:
    from app.gpt_summary import get_gpt_tailored_summary_with_retry
    GPT_SUMMARY_AVAILABLE = True
    logger.info("[STABLE_SCORER] GPT summary module loaded successfully")
except ImportError as e:
    GPT_SUMMARY_AVAILABLE = False
    logger.warning(f"[STABLE_SCORER] GPT summary module not available: {e}")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory cache for responses
response_cache = {}

# Persistent cache file for deterministic results across server restarts
CACHE_FILE = "thumbnail_cache.json"

def load_persistent_cache():
    """Load cache from disk"""
    try:
        import json
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_persistent_cache():
    """Save cache to disk"""
    try:
        import json
        with open(CACHE_FILE, 'w') as f:
            json.dump(response_cache, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# Load existing cache on startup
response_cache = load_persistent_cache()
logger.info(f"[CACHE] Loaded {len(response_cache)} cached results from disk")

# Usage tracking for cost control
usage_tracker = {
    "daily_calls": 0,
    "daily_limit": 100,  # Free tier: 100 calls per day
    "last_reset": datetime.now().date(),
    "total_calls": 0
}

# CTR benchmark data by niche (based on YouTube performance research)
CTR_BENCHMARKS = {
    'gaming': {
        'poor': 1.5,
        'average': 3.0,
        'good': 5.0,
        'excellent': 7.0
    },
    'business': {
        'poor': 1.0,
        'average': 2.0,
        'good': 3.5,
        'excellent': 5.5
    },
    'tech': {
        'poor': 1.2,
        'average': 2.5,
        'good': 4.0,
        'excellent': 6.0
    },
    'food': {
        'poor': 1.8,
        'average': 3.5,
        'good': 5.5,
        'excellent': 8.0
    },
    'fitness': {
        'poor': 1.5,
        'average': 3.0,
        'good': 5.0,
        'excellent': 7.5
    },
    'education': {
        'poor': 1.0,
        'average': 2.0,
        'good': 3.5,
        'excellent': 5.0
    },
    'entertainment': {
        'poor': 2.0,
        'average': 4.0,
        'good': 6.5,
        'excellent': 9.0
    },
    'travel': {
        'poor': 1.5,
        'average': 3.0,
        'good': 5.0,
        'excellent': 7.0
    },
    'music': {
        'poor': 2.0,
        'average': 4.0,
        'good': 6.0,
        'excellent': 8.5
    },
    'general': {
        'poor': 1.5,
        'average': 2.5,
        'good': 4.0,
        'excellent': 6.0
    }
}

# Niche-specific contexts for GPT-4 Vision
NICHE_CONTEXTS = {
    "gaming": "high-energy gameplay, bright colors, action shots, recognizable characters, competitive gaming elements, click-worthy excitement",
    "business": "professional appearance, clean backgrounds, confident expressions, success indicators, corporate aesthetics, trust-building elements",
    "tech": "clean design, modern aesthetics, gadgets/devices, professional lighting, innovation-focused, sleek presentation",
    "food": "appetizing presentation, warm lighting, close-up shots, vibrant colors, mouth-watering appeal, hunger-inducing imagery",
    "fitness": "energetic poses, transformation shots, bold motivational text, gym environments, athletic wear, inspiring visuals",
    "education": "clear, trustworthy presentation, organized layouts, approachable instructors, educational materials, learning-focused",
    "entertainment": "expressive faces, dramatic lighting, emotion-driven content, entertainment value, engaging visuals, personality-driven",
    "travel": "breathtaking scenery, aspirational destinations, adventure elements, wanderlust-inducing imagery, dream-worthy locations",
    "music": "artistic expression, genre-appropriate aesthetics, musical instruments, performance shots, creative vibes, musical energy",
    "general": "clear focal point, broad appeal, high readability, emotional connection, universal understanding, click-worthy elements"
}

def check_usage_limits() -> Dict[str, Any]:
    """
    Check if we're within usage limits to control costs
    Returns status and remaining calls
    """
    today = datetime.now().date()
    
    # Reset daily counter if new day
    if usage_tracker["last_reset"] != today:
        usage_tracker["daily_calls"] = 0
        usage_tracker["last_reset"] = today
        logger.info(f"[USAGE] Daily limit reset. New day: {today}")
    
    remaining_calls = usage_tracker["daily_limit"] - usage_tracker["daily_calls"]
    
    if remaining_calls <= 0:
        logger.warning(f"[USAGE] Daily limit exceeded! Calls: {usage_tracker['daily_calls']}/{usage_tracker['daily_limit']}")
        return {
            "within_limits": False,
            "remaining_calls": 0,
            "daily_calls": usage_tracker["daily_calls"],
            "daily_limit": usage_tracker["daily_limit"]
        }
    
    return {
        "within_limits": True,
        "remaining_calls": remaining_calls,
        "daily_calls": usage_tracker["daily_calls"],
        "daily_limit": usage_tracker["daily_limit"]
    }

def increment_usage():
    """Increment usage counter"""
    usage_tracker["daily_calls"] += 1
    usage_tracker["total_calls"] += 1
    logger.info(f"[USAGE] Call #{usage_tracker['daily_calls']}/{usage_tracker['daily_limit']} (Total: {usage_tracker['total_calls']})")

def get_image_hash(image_bytes: bytes) -> str:
    """Generate deterministic hash from image bytes"""
    return hashlib.sha256(image_bytes).hexdigest()[:16]

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string for API"""
    return base64.b64encode(image_bytes).decode('utf-8')

def get_gpt_rubric(image_bytes: bytes, title: str, niche: str) -> Dict[str, Any]:
    """
    Get GPT-4 Vision rubric analysis with strict JSON schema
    Returns deterministic scores 0-5 for each dimension
    """
    # Check usage limits first
    usage_status = check_usage_limits()
    if not usage_status["within_limits"]:
        logger.warning(f"[GPT-4] Daily limit exceeded, using fallback. Calls: {usage_status['daily_calls']}/{usage_status['daily_limit']}")
        return {
            "visual_appeal": 3,
            "subject_prominence": 3,
            "emotion": 3,
            "text_readability": 3,  # Use 3 instead of 0 for fallback
            "color_contrast": 3,
            "title_alignment": 2,  # Use 2 instead of 3 for more realistic fallback
            "niche_relevance": 3,  # Neutral fallback - no niche penalty
            "notes": f"Daily limit exceeded ({usage_status['daily_calls']}/{usage_status['daily_limit']} calls). Using fallback scoring."
        }
    
    try:
        # Increment usage counter
        increment_usage()
        
        base64_image = encode_image_to_base64(image_bytes)
        niche_context = NICHE_CONTEXTS.get(niche, NICHE_CONTEXTS["general"])
        
        prompt = f"""Analyze this YouTube thumbnail for CLICK OPTIMIZATION in the {niche} niche.

Your goal: Rate how likely this thumbnail is to get CLICKS on YouTube.

Rate each dimension 0-5 (integers only):
- 0: Very poor click potential, 1: Poor click potential, 2: Below average, 3: Average, 4: Good click potential, 5: Excellent click potential

YOUTUBE CLICK OPTIMIZATION CRITERIA:

1. VISUAL APPEAL (0-5):
   - Does it grab attention instantly? Bright colors, high contrast, eye-catching elements?
   - Would it stand out in YouTube's crowded sidebar/search results?
   - Does it look professional and polished?

2. SUBJECT PROMINENCE (0-5):
   - Is the main subject (person, object, scene) clearly visible and prominent?
   - Does it fill enough of the frame to be recognizable at thumbnail size?
   - Is there a clear focal point that draws the eye?

3. EMOTION (0-5):
   - Does it evoke curiosity, excitement, surprise, or strong emotion?
   - Would viewers feel compelled to click to learn more?
   - Does it create an emotional connection or reaction?

4. TEXT READABILITY (0-5):
   - Can any text be read clearly at YouTube thumbnail size (especially on mobile)?
   - Is text large enough, high contrast, and positioned well?
   - Does text add value or create intrigue?

5. COLOR CONTRAST (0-5):
   - High contrast colors that pop against YouTube's interface?
   - Colors that work well for the {niche} niche?
   - Does it avoid being too dark or washed out?

6. TITLE ALIGNMENT (0-5):
   - Does thumbnail perfectly match the title "{title}"?
   - Would viewers feel misled or satisfied when they click?
   - Does it deliver on the promise made by the title?

7. NICHE RELEVANCE (0-5):
   - Perfectly matches {niche} content expectations?
   - Would {niche} viewers recognize and trust this thumbnail style?
   - Follows successful {niche} thumbnail patterns?

YOUTUBE-SPECIFIC CONSIDERATIONS:
- Mobile-first: Most YouTube viewing is on phones
- Competition: Must stand out among similar videos
- Algorithm-friendly: Clear, engaging, relevant content
- Click-through rate optimization: Creates curiosity without being misleading

Return ONLY this JSON format:
{{
    "visual_appeal": 0,
    "subject_prominence": 0,
    "emotion": 0,
    "text_readability": 0,
    "color_contrast": 0,
    "title_alignment": 0,
    "niche_relevance": 0,
    "notes": "YouTube click optimization analysis"
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,  # Low temperature for consistency
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        )
        
        # Parse JSON response
        content = response.choices[0].message.content.strip()
        
        # Debug logging
        logger.info(f"[GPT-4] Raw response length: {len(content)}")
        logger.info(f"[GPT-4] Raw response preview: {content[:100]}...")
        
        if not content:
            logger.error("[GPT-4] Empty response from API")
            raise ValueError("Empty response from GPT-4 Vision API")
        
        # Clean up markdown formatting
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        # Try to parse JSON
        try:
            rubric = json.loads(content)
        except json.JSONDecodeError as json_err:
            logger.error(f"[GPT-4] JSON parse error: {json_err}")
            logger.error(f"[GPT-4] Content that failed to parse: {content}")
            raise ValueError(f"Invalid JSON response: {json_err}")
        
        # Validate and clamp all rubric values to 0-5
        for key in ["visual_appeal", "subject_prominence", "emotion", "text_readability", "color_contrast", "title_alignment"]:
            if key in rubric:
                rubric[key] = max(0, min(5, int(rubric[key])))
            else:
                rubric[key] = 3  # Safe default
        
        return rubric
        
    except Exception as e:
        logger.error(f"GPT-4 Vision rubric failed: {e}")
        # Return safe default rubric
        return {
            "visual_appeal": 3,
            "subject_prominence": 3,
            "emotion": 3,
            "text_readability": 3,  # Use 3 instead of 0 for fallback
            "color_contrast": 3,
            "title_alignment": 2,  # Use 2 instead of 3 for more realistic fallback
            "niche_relevance": 3,  # Neutral fallback - no niche penalty
            "notes": "Fallback scoring due to API error"
        }

def text_clarity(image_bytes: bytes) -> Tuple[int, str]:
    """
    Deterministic text clarity score 0-100 + extracted OCR text
    Based on OCR confidence + luminance contrast of text areas
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        
        # Get OCR data
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Extract OCR text with better configuration
        try:
            # Try multiple OCR configurations for better text extraction
            ocr_configs = [
                '--psm 8',  # Single word
                '--psm 7',  # Single text line
                '--psm 6',  # Single uniform block of text
                '--psm 3',  # Fully automatic page segmentation
            ]
            
            ocr_text = ""
            for config in ocr_configs:
                try:
                    text = pytesseract.image_to_string(image, config=config).strip()
                    if text and len(text) > 3 and text.replace(' ', '').isalnum():
                        ocr_text = text
                        logger.info(f"[OCR_DEBUG] Extracted text with {config}: '{ocr_text}' (length={len(ocr_text)})")
                        break
                except:
                    continue
            
            # If no meaningful text found, try without config
            if not ocr_text:
                ocr_text = pytesseract.image_to_string(image).strip()
                if ocr_text and len(ocr_text) > 3:
                    logger.info(f"[OCR_DEBUG] Extracted text (no config): '{ocr_text}' (length={len(ocr_text)})")
                else:
                    logger.info(f"[OCR_DEBUG] No meaningful text found")
                    ocr_text = ""
                    
        except Exception as e:
            logger.error(f"[OCR_DEBUG] OCR extraction failed: {e}")
            ocr_text = ""
        
        # Calculate average confidence for detected text
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        
        if not confidences:
            return 50, ocr_text  # No text detected - neutral score, but return extracted text
        
        avg_confidence = sum(confidences) / len(confidences)
        
        # Calculate luminance contrast for text regions
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        text_regions = []
        
        for i, conf in enumerate(ocr_data['conf']):
            if int(conf) > 30:  # Only consider confident detections
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                if w > 10 and h > 5:  # Valid text region
                    region = gray[y:y+h, x:x+w]
                    if region.size > 0:
                        text_regions.append(region)
        
        # Calculate contrast score
        if text_regions:
            contrasts = []
            for region in text_regions:
                if region.size > 0:
                    # Simple contrast measure: std deviation of pixel values
                    contrast = np.std(region)
                    contrasts.append(contrast)
            
            if contrasts:
                avg_contrast = sum(contrasts) / len(contrasts)
                contrast_score = min(100, avg_contrast * 2)  # Scale to 0-100
            else:
                contrast_score = 50
        else:
            contrast_score = 50
        
        # Combine confidence and contrast
        final_score = (avg_confidence * 0.6) + (contrast_score * 0.4)
        return int(max(0, min(100, final_score))), ocr_text
        
    except Exception as e:
        logger.error(f"Text clarity calculation failed: {e}")
        return 50, ""  # Safe default with empty text

def color_contrast(image_bytes: bytes) -> int:
    """
    Deterministic color contrast score 0-100
    Based on HSV spread and WCAG-like contrast proxy
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Calculate HSV spread (variance)
        h_spread = np.var(h) if h.size > 0 else 0
        s_spread = np.var(s) if s.size > 0 else 0
        v_spread = np.var(v) if v.size > 0 else 0
        
        # Normalize spreads to 0-100
        h_score = min(100, h_spread / 100)
        s_score = min(100, s_spread / 100)
        v_score = min(100, v_spread / 100)
        
        # Weighted combination
        contrast_score = (h_score * 0.3) + (s_score * 0.4) + (v_score * 0.3)
        return int(max(0, min(100, contrast_score)))
        
    except Exception as e:
        logger.error(f"Color contrast calculation failed: {e}")
        return 50  # Safe default

def subject_size(image_bytes: bytes) -> Tuple[int, List[Dict]]:
    """
    Deterministic subject size score 0-100 + face detection data
    Uses MediaPipe face detection or OpenCV saliency fallback
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        
        # Try MediaPipe face detection first
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            mp_drawing = mp.solutions.drawing_utils
            
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(image_array)
                
                if results.detections:
                    # Calculate total face area as percentage of image
                    total_face_area = 0
                    face_boxes = []
                    
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        face_area = bbox.width * bbox.height
                        total_face_area += face_area
                        
                        # Store face detection data
                        face_boxes.append({
                            "bbox": [
                                int(bbox.xmin * width),
                                int(bbox.ymin * height),
                                int(bbox.width * width),
                                int(bbox.height * height)
                            ],
                            "confidence": detection.score[0],
                            "emotion": "Detected",  # Face detected but emotion analysis not available
                            "age": "Detected",
                            "gender": "Detected"
                        })
                    
                    # Convert to percentage and scale to 0-100
                    face_percentage = total_face_area * 100
                    score = int(max(0, min(100, face_percentage * 2)))  # Scale factor
                    return score, face_boxes
                else:
                    # No faces detected - use saliency fallback
                    score = subject_size_saliency_fallback(image_array)
                    return score, []
                    
        except ImportError:
            # MediaPipe not available - use saliency fallback
            score = subject_size_saliency_fallback(image_array)
            return score, []
            
    except Exception as e:
        logger.error(f"Subject size calculation failed: {e}")
        return 50, []  # Safe default with empty face list

def subject_size_saliency_fallback(image_array: np.ndarray) -> int:
    """
    Fallback subject size calculation using OpenCV saliency
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Simple edge detection for saliency proxy
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density as proxy for subject prominence
        edge_density = np.sum(edges > 0) / edges.size
        
        # Convert to percentage (0-100)
        salient_percentage = edge_density * 100
        
        return int(max(0, min(100, salient_percentage * 2)))  # Scale factor
        
    except Exception as e:
        logger.error(f"Saliency fallback failed: {e}")
        return 50

def saturation_energy(image_bytes: bytes) -> int:
    """
    Deterministic saturation energy score 0-100
    Based on mean saturation + edge density
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]  # Saturation channel
        
        # Calculate mean saturation
        mean_saturation = np.mean(s)
        
        # Calculate edge density
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine saturation and edge density
        saturation_score = mean_saturation * 0.6
        edge_score = edge_density * 100 * 0.4
        
        final_score = saturation_score + edge_score
        return int(max(0, min(100, final_score)))
        
    except Exception as e:
        logger.error(f"Saturation energy calculation failed: {e}")
        return 50  # Safe default

def calculate_numeric_core(image_bytes: bytes) -> Dict[str, Any]:
    """
    Calculate deterministic numeric core score
    Returns individual scores, weighted combination, and additional data
    """
    text_score, ocr_text = text_clarity(image_bytes)
    contrast = color_contrast(image_bytes)
    subject_score, face_boxes = subject_size(image_bytes)
    saturation = saturation_energy(image_bytes)
    
    # Weighted combination
    core_score = round(
        0.30 * text_score +
        0.30 * contrast +
        0.20 * subject_score +
        0.20 * saturation
    )
    
    return {
        "text_clarity": text_score,
        "color_contrast": contrast,
        "subject_size": subject_score,
        "saturation_energy": saturation,
        "core_score": core_score,
        "ocr_text": ocr_text,
        "face_boxes": face_boxes
    }

def calculate_rubric_score(rubric: Dict[str, Any]) -> float:
    """
    Calculate rubric score from GPT-4 Vision analysis
    Maps 0-5 scale to 0-100 and applies title alignment modifier
    """
    # Map rubric dimensions (0-5) to 0-100 scale
    dimensions = [
        rubric.get("visual_appeal", 3),
        rubric.get("subject_prominence", 3),
        rubric.get("emotion", 3),
        rubric.get("text_readability", 3),
        rubric.get("color_contrast", 3)
    ]
    
    # Average of dimensions
    base_rubric_score = (sum(dimensions) / len(dimensions)) * 20  # Convert 0-5 to 0-100
    
    # Apply title alignment modifier
    title_alignment = rubric.get("title_alignment", 3)
    modifier = 0.9 + (0.04 * title_alignment)  # Range: 0.9 to 1.1
    rubric_score = base_rubric_score * modifier
    
    # Apply niche relevance penalty (CRITICAL!)
    niche_relevance = rubric.get("niche_relevance", 3)
    original_score = rubric_score
    
    if niche_relevance <= 1:
        # Heavy penalty for wrong niche (gym thumbnail for business content)
        rubric_score = rubric_score * 0.2  # Reduce to 20% of original score
        logger.warning(f"[NICHE] Heavy penalty applied: {niche_relevance}/5 -> {original_score:.1f} -> {rubric_score:.1f} (80% reduction)")
    elif niche_relevance <= 2:
        # Moderate penalty for poor niche match
        rubric_score = rubric_score * 0.5  # Reduce to 50% of original score
        logger.info(f"[NICHE] Moderate penalty applied: {niche_relevance}/5 -> {original_score:.1f} -> {rubric_score:.1f} (50% reduction)")
    elif niche_relevance <= 3:
        # Slight penalty for average niche match
        rubric_score = rubric_score * 0.8  # Reduce to 80% of original score
        logger.info(f"[NICHE] Slight penalty applied: {niche_relevance}/5 -> {original_score:.1f} -> {rubric_score:.1f} (20% reduction)")
    else:
        # niche_relevance >= 4 gets no penalty (good match)
        logger.info(f"[NICHE] Good niche match: {niche_relevance}/5 -> No penalty applied")
    
    return round(rubric_score, 1)

def scale_to_realistic_range(raw_score: float) -> float:
    """
    Maps internal 0-100 score to realistic 30-95 range with better distribution.
    Uses non-linear scaling to spread scores more naturally and feel more rewarding.
    
    Args:
        raw_score: Internal score from 0-100
    
    Returns:
        Final score: 30-95 range
    """
    
    # Apply amplified non-linear scaling for better user experience
    if raw_score < 35:
        # Poor range: 0-35 → 30-50
        return 30 + (raw_score * 0.57)  # Slightly more generous
        
    elif raw_score < 55:
        # Fair range: 35-55 → 50-68
        return 50 + ((raw_score - 35) * 0.9)  # More generous spread
        
    elif raw_score < 70:
        # Good range: 55-70 → 68-78
        return 68 + ((raw_score - 55) * 0.67)
        
    elif raw_score < 85:
        # Strong range: 70-85 → 78-88
        return 78 + ((raw_score - 70) * 0.67)
        
    else:
        # Excellent range: 85-100 → 88-95
        return 88 + ((raw_score - 85) * 0.47)


def get_score_tier(score: float) -> Tuple[str, str, str]:
    """
    Return tier label and icon based on final score.
    
    Args:
        score: Final score (30-95)
    
    Returns:
        tuple: (tier_label, tier_icon, tier_color)
    """
    
    if score >= 86:
        return 'Excellent', '🟢', 'green'
    elif score >= 76:
        return 'Strong', '🟢', 'green'
    elif score >= 66:
        return 'Good', '🟡', 'yellow'
    elif score >= 51:
        return 'Fair', '🟠', 'orange'
    else:
        return 'Needs Work', '🔴', 'red'


def calculate_ctr_prediction(score: float, niche: str) -> Dict[str, Any]:
    """
    Calculate CTR prediction with contextual information.
    
    Args:
        score: Final thumbnail score (30-95)
        niche: Content category
    
    Returns:
        dict: CTR prediction with context including performance tier and comparison
    """
    
    benchmarks = CTR_BENCHMARKS.get(niche, CTR_BENCHMARKS['general'])
    
    # Map score to CTR range and performance tier
    if score >= 86:
        ctr_min = benchmarks['excellent'] * 0.9
        ctr_max = benchmarks['excellent'] * 1.1
        performance = 'Excellent'
        comparison = f'Well above average for {niche}'
        stars = 5
        
    elif score >= 76:
        ctr_min = benchmarks['good'] * 0.9
        ctr_max = benchmarks['good'] * 1.1
        performance = 'Strong'
        comparison = f'Above average for {niche}'
        stars = 4
        
    elif score >= 66:
        ctr_min = benchmarks['average'] * 0.9
        ctr_max = benchmarks['average'] * 1.1
        performance = 'Good'
        comparison = f'Average for {niche}'
        stars = 3
        
    elif score >= 51:
        ctr_min = benchmarks['poor'] * 0.9
        ctr_max = benchmarks['average'] * 0.8
        performance = 'Fair'
        comparison = f'Below average for {niche}'
        stars = 2
        
    else:
        ctr_min = benchmarks['poor'] * 0.7
        ctr_max = benchmarks['poor'] * 1.0
        performance = 'Needs Work'
        comparison = f'Significantly below average for {niche}'
        stars = 1
    
    return {
        'ctr_min': round(ctr_min, 1),
        'ctr_max': round(ctr_max, 1),
        'ctr_range': f"{round(ctr_min, 1)}-{round(ctr_max, 1)}%",
        'performance': performance,
        'comparison': comparison,
        'benchmark_average': benchmarks['average'],
        'stars': stars,
        'confidence': 85  # Standard confidence level for CTR predictions
    }


def calculate_confidence(rubric_score: float, numeric_core: int, rubric: Dict[str, Any]) -> str:
    """
    Calculate confidence level based on agreement and consistency
    """
    # Agreement between rubric and numeric core
    agreement = 100 - min(100, abs(rubric_score - numeric_core))
    
    # Spread of rubric dimensions (consistency)
    dimensions = [
        rubric.get("visual_appeal", 3) * 20,
        rubric.get("subject_prominence", 3) * 20,
        rubric.get("emotion", 3) * 20,
        rubric.get("text_readability", 3) * 20,
        rubric.get("color_contrast", 3) * 20
    ]
    spread = statistics.stdev(dimensions) if len(dimensions) > 1 else 0
    
    # Determine confidence level
    if agreement > 70 and spread < 18:
        return "high"
    elif agreement > 50:
        return "medium"
    else:
        return "low"

def score_thumbnail_stable(image_bytes: bytes, title: str, niche: str) -> Dict[str, Any]:
    """
    Main scoring function - deterministic and cacheable
    Returns complete scoring breakdown
    """
    # Generate deterministic hash
    hash_id = get_image_hash(image_bytes)
    
    # Check cache first
    cache_key = f"{hash_id}_{title}_{niche}_v1.0-gpt4rubric-core"
    if cache_key in response_cache:
        logger.info(f"[Cache] hit for hash={hash_id[:8]}...")
        return response_cache[cache_key]
    
    logger.info(f"[Cache] miss for hash={hash_id[:8]}...")
    
    # Get GPT-4 Vision rubric
    rubric = get_gpt_rubric(image_bytes, title, niche)
    rubric_score = calculate_rubric_score(rubric)
    
    logger.info(f"[Rubric] scores: {rubric} -> rubric_score={rubric_score}")
    
    # Calculate numeric core
    numeric_core_data = calculate_numeric_core(image_bytes)
    numeric_core = numeric_core_data["core_score"]
    
    logger.info(f"[Core] text={numeric_core_data['text_clarity']} contrast={numeric_core_data['color_contrast']} subject={numeric_core_data['subject_size']} sat={numeric_core_data['saturation_energy']} -> core={numeric_core}")
    
    # Blend scores: 55% rubric, 45% numeric core
    final_score = 0.55 * rubric_score + 0.45 * numeric_core
    
    # Apply new realistic scaling for better user experience
    # Map scores to 30-95 range with improved distribution
    final_score = scale_to_realistic_range(final_score)
    
    # Final clamp to reasonable range
    final_score = min(95, max(30, round(final_score, 1)))
    
    # Calculate confidence, tier, and CTR prediction
    confidence = calculate_confidence(rubric_score, numeric_core, rubric)
    tier_label, tier_icon, tier_color = get_score_tier(final_score)
    ctr_prediction = calculate_ctr_prediction(final_score, niche)
    
    logger.info(f"[Blend] final={final_score} tier={tier_label} confidence={confidence} CTR={ctr_prediction['ctr_range']}")
    
    # Generate GPT-4 Vision tailored summary if available
    gpt_summary = None
    if GPT_SUMMARY_AVAILABLE:
        try:
            # Prepare comprehensive metrics for GPT summary
            metrics = {
                "title": title,
                "niche": niche,
                "ocr_text": numeric_core_data.get("ocr_text", ""),
                "text_boxes": [],  # Could be enhanced with OCR bounding boxes
                "faces": numeric_core_data.get("face_boxes", [])[:3],  # Use actual face detection data
                "subject_pct_estimate": numeric_core_data.get("subject_size", 0) / 100,
                "saliency_pct_main": numeric_core_data.get("subject_size", 0) / 100,  # Use subject size as proxy
                "avg_saturation": numeric_core_data.get("saturation_energy", 50) / 100,
                "dominant_colors": [],  # Could be enhanced with color analysis
                "rule_of_thirds_hits": 2,  # Could be calculated from face/subject positions
                "library_trend_match": 0.7,  # Estimate based on niche
                # Additional technical data for more specific analysis
                "text_clarity_score": numeric_core_data.get("text_clarity", 50),
                "color_contrast_score": numeric_core_data.get("color_contrast", 50),
                "final_score": final_score,
                "rubric_scores": {
                    "visual_appeal": rubric.get("visual_appeal", 3),
                    "subject_prominence": rubric.get("subject_prominence", 3),
                    "emotion": rubric.get("emotion", 3),
                    "text_readability": rubric.get("text_readability", 3),
                    "color_contrast": rubric.get("color_contrast", 3),
                    "title_alignment": rubric.get("title_alignment", 3),
                    "niche_relevance": rubric.get("niche_relevance", 3)
                }
            }
            
            logger.info(f"[GPT-SUMMARY] Generating tailored summary for thumbnail")
            gpt_summary = get_gpt_tailored_summary_with_retry(image_bytes, metrics)
            logger.info(f"[GPT-SUMMARY] Generated summary: {gpt_summary.get('winner_summary', '')[:50]}...")
        except Exception as e:
            logger.warning(f"[GPT-SUMMARY] Failed to generate summary: {e}")
            gpt_summary = None
    
    # Build response
    response = {
        "thumbscore": final_score,
        "confidence": confidence,
        "tier": tier_label,
        "tier_icon": tier_icon,
        "tier_color": tier_color,
        "ctr_prediction": ctr_prediction,  # Add CTR prediction with context
        "score_version": "v1.2-ctr-context",
        "hash": hash_id,
        "niche": niche,
        "rubric": {
            **rubric,
            "rubric_score": rubric_score
        },
        "numeric_core": numeric_core_data,
        "calibration": {"min": 30, "max": 95},
        "gpt_summary": gpt_summary  # Add GPT summary to response
    }
    
    # Cache response
    response_cache[cache_key] = response
    save_persistent_cache()  # Save to disk for persistence
    
    return response

def compare_thumbnails_stable(thumbnails: List[Dict[str, Any]], title: str, niche: str) -> Dict[str, Any]:
    """
    Score multiple thumbnails and determine winner with duplicate detection
    """
    results = []
    hash_counts = {}
    processed_hashes = {}  # Track which thumbnails have identical content
    
    logger.info(f"[YouTube] Analyzing thumbnail for {niche} niche - Click optimization focus")
    
    # Score each thumbnail and track duplicates
    for thumb in thumbnails:
        # Generate hash from image data only
        image_hash = get_image_hash(thumb["image_data"])
        
        # Check if we've already processed this exact image
        if image_hash in processed_hashes:
            logger.info(f"[STABLE_SCORER] Duplicate detected: {thumb['id']} identical to {processed_hashes[image_hash]}")
            # Use cached result for identical image
            cached_result = processed_hashes[image_hash].copy()
            cached_result["id"] = thumb["id"]
            cached_result["duplicate_of"] = processed_hashes[image_hash]["id"]
            results.append(cached_result)
            hash_counts[image_hash] = hash_counts.get(image_hash, 0) + 1
            continue
        
        # Process new unique image
        score_data = score_thumbnail_stable(thumb["image_data"], title, niche)
        score_data["id"] = thumb["id"]
        
        # Store for duplicate detection
        processed_hashes[image_hash] = score_data
        hash_counts[image_hash] = hash_counts.get(image_hash, 0) + 1
        results.append(score_data)
        logger.info(f"[STABLE_SCORER] {thumb['id']}: {score_data['thumbscore']}/100")
    
    # Determine winner (highest score)
    if results:
        max_score = max(result["thumbscore"] for result in results)
        winner_indices = [i for i, result in enumerate(results) if result["thumbscore"] == max_score]
        winner_index = winner_indices[0]  # Take first if tie
        
        # Mark winner
        results[winner_index]["is_winner"] = True
        winner_id = results[winner_index]["id"]
        
        logger.info(f"[STABLE_SCORER] Winner: {winner_id} with {max_score}/100")
    else:
        winner_id = "none"
    
    return {
        "winner_id": winner_id,
        "thumbnails": results,
        "explanation": f"Winner selected based on YouTube click optimization analysis. Thumbnail {winner_id} scored highest for click-through rate potential in the {niche} niche.",
        "niche": niche,
        "metadata": {
            "scoring_version": "v1.0-gpt4rubric-core",
            "total_thumbnails": len(thumbnails),
            "duplicates_detected": sum(1 for count in hash_counts.values() if count > 1)
        }
    }

def get_preview_score(image_bytes: bytes, title: str, niche: str) -> Dict[str, Any]:
    """
    Generate cheap preview score using only deterministic numeric core
    No GPT Vision API calls - completely free
    """
    try:
        logger.info(f"[PREVIEW] Generating preview score for niche: {niche}")
        
        # Calculate numeric core only (no GPT Vision)
        numeric_core_data = calculate_numeric_core(image_bytes)
        
        # Generate preview score (numeric core only)
        preview_score = numeric_core_data["core_score"]
        
        # Apply same realistic scaling as main scoring
        preview_score = scale_to_realistic_range(preview_score)
        
        # Final clamp to reasonable range
        preview_score = min(95, max(30, round(preview_score, 1)))
        
        # Get tier information
        tier_label, tier_icon, tier_color = get_score_tier(preview_score)
        
        # Build preview response
        response = {
            "preview_score": preview_score,
            "tier": tier_label,
            "tier_icon": tier_icon,
            "tier_color": tier_color,
            "score_version": "v1.1-preview",
            "niche": niche,
            "numeric_core": numeric_core_data,
            "capable_of_full_ai": True,
            "preview_note": "This is a preview using numeric analysis only. Run full AI analysis for detailed insights.",
            "calibration": {"min": 30, "max": 95}
        }
        
        logger.info(f"[PREVIEW] Generated preview score: {preview_score}/100")
        return response
        
    except Exception as e:
        logger.error(f"[PREVIEW] Error generating preview score: {e}")
        # Return safe fallback
        return {
            "preview_score": 50,
            "score_version": "v1.0-preview-error",
            "niche": niche,
            "numeric_core": {
                "text_clarity": 50,
                "color_contrast": 50,
                "subject_size": 50,
                "saturation_energy": 50,
                "core_score": 50
            },
            "capable_of_full_ai": True,
            "preview_note": "Preview analysis failed, using fallback score.",
            "calibration": {"min": 30, "max": 95},
            "error": str(e)
        }
