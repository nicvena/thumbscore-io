"""
GPT-4 Vision helper for generating tailored, evidence-based thumbnail summaries.
Provides strict JSON outputs with concrete insights about actual thumbnail elements.
"""

import json
import os
import base64
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client (only if API key is available)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Example schema for reference
SCHEMA_EXAMPLE = {
    "winner_summary": "string, 1–2 sentences, tailored and specific",
    "insights": [
        {
            "label": "string, <= 5 words (e.g., 'Readable headline')",
            "evidence": "string, concrete & image-specific (e.g., \"WHITE 'KETO DIET' text, 92% contrast, top-right\")"
        }
    ]
}

# Banned generic phrases that indicate non-specific analysis
BANNED_GENERIC = [
    "eye-catching", "good colors", "nice contrast", "pops", "attractive", "appealing", 
    "clean look", "modern look", "stands out", "looks great", "effective", "engaging",
    "compelling", "striking", "vibrant", "bold", "dynamic", "professional", "polished",
    "high quality", "well designed", "catchy", "attention-grabbing", "visually appealing"
]

def _looks_generic(text: str) -> bool:
    """Check if text contains generic, non-specific language."""
    if not text:
        return True
    low = text.lower()
    return any(phrase in low for phrase in BANNED_GENERIC)

def _validate_json_schema(data: Dict[str, Any]) -> bool:
    """Validate that the JSON response matches our expected schema."""
    if not isinstance(data, dict):
        return False
    
    # Check required fields
    if "winner_summary" not in data or "insights" not in data:
        return False
    
    # Validate winner_summary
    summary = data.get("winner_summary", "")
    if not isinstance(summary, str) or len(summary.strip()) < 10:
        return False
    
    # Validate insights
    insights = data.get("insights", [])
    if not isinstance(insights, list) or len(insights) == 0:
        return False
    
    for insight in insights:
        if not isinstance(insight, dict):
            return False
        if "label" not in insight or "evidence" not in insight:
            return False
        if not isinstance(insight["label"], str) or not isinstance(insight["evidence"], str):
            return False
        if len(insight["label"].strip()) < 2 or len(insight["evidence"].strip()) < 10:
            return False
    
    return True

def _clean_insights(insights: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Clean and validate insights, removing generic content."""
    cleaned = []
    for insight in insights[:3]:  # Max 3 insights
        if not isinstance(insight, dict):
            continue
            
        label = (insight.get("label") or "").strip()[:40]
        evidence = (insight.get("evidence") or "").strip()
        
        # Skip if too short, generic, or missing required fields
        if (len(label) < 2 or len(evidence) < 10 or 
            _looks_generic(label) or _looks_generic(evidence)):
            continue
            
        cleaned.append({
            "label": label,
            "evidence": evidence
        })
    
    return cleaned

def get_gpt_tailored_summary(image_bytes: bytes, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a tailored, evidence-based summary for a thumbnail using GPT-4 Vision.
    
    Args:
        image_bytes: Raw image bytes (PNG/JPEG)
        metrics: Dictionary containing precomputed thumbnail metrics
        
    Returns:
        Dictionary with winner_summary and insights
    """
    if not client or not openai_api_key:
        return {
            "winner_summary": "AI analysis unavailable - OpenAI API not configured.",
            "insights": []
        }
    
    try:
        # Prepare system prompt
        system_prompt = (
            "You are a YouTube thumbnail optimization expert with access to precise visual analysis data. "
            "Return ONLY valid JSON. Be specific and verifiable. "
            "Reference ACTUAL elements you see: exact words, colors, relative sizes (% of frame), layout positions. "
            "No vague phrases, no marketing fluff. Use the provided metrics to support your analysis."
        )
        
        # Prepare metrics context
        metrics_json = json.dumps({
            "title": metrics.get("title", ""),
            "niche": metrics.get("niche", ""),
            "ocr_text": metrics.get("ocr_text", ""),
            "text_boxes": metrics.get("text_boxes", [])[:3],
            "faces": metrics.get("faces", [])[:3],
            "subject_pct_estimate": metrics.get("subject_pct_estimate", 0),
            "saliency_pct_main": metrics.get("saliency_pct_main", 0),
            "avg_saturation": metrics.get("avg_saturation", 0),
            "dominant_colors": metrics.get("dominant_colors", [])[:3],
            "rule_of_thirds_hits": metrics.get("rule_of_thirds_hits", 0),
            "library_trend_match": metrics.get("library_trend_match", 0),
            "requirements": {
                "insights_max": 3,
                "at_least_one_numeric": True,
                "must_reference_real_text_if_present": True
            }
        }, ensure_ascii=False)
        
        # Prepare user prompt
        user_prompt = (
            "Analyze this YouTube thumbnail WITH the provided metrics. "
            "Produce JSON with this exact schema:\n"
            "{\n"
            "  \"winner_summary\": \"1–2 sentences, tailored, no fluff\",\n"
            "  \"insights\": [\n"
            "     {\"label\": \"<=5 words\", \"evidence\": \"specific: text seen, color, % frame, placement\"}\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Cite concrete details you can see (e.g., \"ALL-CAPS 'KETO DIET' in white on red, top-right\").\n"
            "- Include numbers where possible (~% of frame, contrast %, thirds hits).\n"
            "- Max 3 insights. No generic adjectives. No extra keys. JSON only.\n"
            "- If text is present in OCR data, reference it specifically.\n"
            "- Use the dominant colors and composition data provided.\n\n"
            "Metrics Context:\n" + metrics_json
        )
        
        # Convert image to base64 for API
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Detect image format from bytes
        image_format = "jpeg"  # default
        if image_bytes.startswith(b'\x89PNG'):
            image_format = "png"
        elif image_bytes.startswith(b'\xff\xd8\xff'):
            image_format = "jpeg"
        elif image_bytes.startswith(b'GIF'):
            image_format = "gif"
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            image_format = "webp"
        
        # Prepare messages for GPT-4 Vision
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        # Call GPT-4 Vision API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for better vision capabilities
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=800
        )
        
        # Parse response
        try:
            data = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from GPT-4")
        
        # Validate schema
        if not _validate_json_schema(data):
            raise ValueError("Invalid JSON schema from GPT-4")
        
        # Clean insights
        data["insights"] = _clean_insights(data.get("insights", []))
        
        # Final validation
        if _looks_generic(data.get("winner_summary", "")):
            raise ValueError("Generic summary detected")
        
        if len(data["insights"]) == 0:
            raise ValueError("No valid insights generated")
        
        return data
        
    except Exception as e:
        # Fallback minimal response
        print(f"[GPT-SUMMARY] Error generating summary: {e}")
        return {
            "winner_summary": f"Analysis completed with {metrics.get('subject_pct_estimate', 0)*100:.0f}% subject prominence and {metrics.get('rule_of_thirds_hits', 0)}/4 composition points.",
            "insights": [
                {
                    "label": "Subject Size",
                    "evidence": f"Main subject occupies ~{metrics.get('subject_pct_estimate', 0)*100:.0f}% of frame"
                },
                {
                    "label": "Composition",
                    "evidence": f"Uses {metrics.get('rule_of_thirds_hits', 0)}/4 rule-of-thirds intersections"
                }
            ]
        }

def get_gpt_tailored_summary_with_retry(image_bytes: bytes, metrics: Dict[str, Any], max_retries: int = 1) -> Dict[str, Any]:
    """
    Generate GPT summary with one retry attempt on failure.
    
    Args:
        image_bytes: Raw image bytes
        metrics: Thumbnail metrics dictionary
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with winner_summary and insights
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            result = get_gpt_tailored_summary(image_bytes, metrics)
            
            # Additional validation
            if (len(result.get("insights", [])) > 0 and 
                not _looks_generic(result.get("winner_summary", ""))):
                return result
            
            if attempt < max_retries:
                print(f"[GPT-SUMMARY] Attempt {attempt + 1} failed validation, retrying...")
                continue
                
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"[GPT-SUMMARY] Attempt {attempt + 1} failed: {e}, retrying...")
                continue
    
    # If all attempts failed, return fallback
    print(f"[GPT-SUMMARY] All attempts failed, using fallback. Last error: {last_error}")
    return {
        "winner_summary": f"Thumbnail analysis completed using visual metrics: {metrics.get('subject_pct_estimate', 0)*100:.0f}% subject coverage, {metrics.get('avg_saturation', 0)*100:.0f}% color saturation.",
        "insights": [
            {
                "label": "Visual Metrics",
                "evidence": f"Subject prominence: {metrics.get('subject_pct_estimate', 0)*100:.0f}%, Saturation: {metrics.get('avg_saturation', 0)*100:.0f}%"
            },
            {
                "label": "Composition",
                "evidence": f"Rule of thirds alignment: {metrics.get('rule_of_thirds_hits', 0)}/4 intersection points"
            }
        ]
    }
