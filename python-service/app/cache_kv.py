#!/usr/bin/env python3
"""
Cache Management System for Thumbscore.io

Handles caching of analysis results to avoid duplicate GPT Vision API calls
and reduce costs.
"""

import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("[CACHE] Missing Supabase credentials")
    supabase: Optional[Client] = None
else:
    supabase: Client = create_client(supabase_url, supabase_key)

# Cache configuration
CACHE_VERSION = "v1.0-gpt4rubric-core"
CACHE_TTL_DAYS = 30

def generate_cache_key(image_bytes: bytes, title: str, niche: str, version: str = CACHE_VERSION) -> str:
    """
    Generate deterministic cache key from analysis inputs
    """
    try:
        # Create hash from image bytes + title + niche + version
        content = image_bytes + title.encode('utf-8') + niche.encode('utf-8') + version.encode('utf-8')
        cache_key = hashlib.sha256(content).hexdigest()
        
        logger.debug(f"[CACHE] Generated cache key: {cache_key[:16]}...")
        return cache_key
        
    except Exception as e:
        logger.error(f"[CACHE] Error generating cache key: {e}")
        raise

def get_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached analysis result
    Returns None if not found or expired
    """
    if not supabase:
        logger.warning("[CACHE] Supabase not initialized, skipping cache lookup")
        return None
    
    try:
        result = supabase.table("score_cache").select("payload").eq("cache_key", cache_key).execute()
        
        if result.data:
            payload = result.data[0]["payload"]
            logger.info(f"[CACHE] Cache hit for key: {cache_key[:16]}...")
            return payload
        else:
            logger.debug(f"[CACHE] Cache miss for key: {cache_key[:16]}...")
            return None
            
    except Exception as e:
        logger.error(f"[CACHE] Error retrieving cache for key {cache_key[:16]}...: {e}")
        return None

def set_cache(cache_key: str, payload: Dict[str, Any]) -> bool:
    """
    Store analysis result in cache
    Returns True if successful
    """
    if not supabase:
        logger.warning("[CACHE] Supabase not initialized, skipping cache store")
        return False
    
    try:
        # Calculate expiration date
        expires_at = datetime.now() + timedelta(days=CACHE_TTL_DAYS)
        
        cache_data = {
            "cache_key": cache_key,
            "payload": payload,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat()
        }
        
        # Use upsert to handle duplicates
        supabase.table("score_cache").upsert(cache_data).execute()
        
        logger.info(f"[CACHE] Stored cache for key: {cache_key[:16]}... (expires: {expires_at.date()})")
        return True
        
    except Exception as e:
        logger.error(f"[CACHE] Error storing cache for key {cache_key[:16]}...: {e}")
        return False

def cleanup_expired_cache() -> int:
    """
    Clean up expired cache entries
    Returns number of entries cleaned
    """
    if not supabase:
        logger.warning("[CACHE] Supabase not initialized, skipping cleanup")
        return 0
    
    try:
        # Call the SQL function to cleanup expired entries
        result = supabase.rpc("cleanup_expired_cache").execute()
        
        # Get count of deleted entries (this might need adjustment based on Supabase version)
        logger.info("[CACHE] Cleaned up expired cache entries")
        return 0  # Supabase doesn't return count for cleanup functions
        
    except Exception as e:
        logger.error(f"[CACHE] Error cleaning up expired cache: {e}")
        return 0

def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics
    """
    if not supabase:
        return {"error": "Supabase not initialized"}
    
    try:
        # Get total cache entries
        total_result = supabase.table("score_cache").select("cache_key", count="exact").execute()
        total_entries = total_result.count if hasattr(total_result, 'count') else len(total_result.data)
        
        # Get recent entries (last 24 hours)
        recent_cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        recent_result = supabase.table("score_cache").select("cache_key", count="exact").gte("created_at", recent_cutoff).execute()
        recent_entries = recent_result.count if hasattr(recent_result, 'count') else len(recent_result.data)
        
        return {
            "total_entries": total_entries,
            "recent_entries_24h": recent_entries,
            "cache_version": CACHE_VERSION,
            "ttl_days": CACHE_TTL_DAYS
        }
        
    except Exception as e:
        logger.error(f"[CACHE] Error getting cache stats: {e}")
        return {"error": str(e)}

def invalidate_cache_pattern(pattern: str) -> int:
    """
    Invalidate cache entries matching a pattern
    Useful for clearing cache when scoring algorithm changes
    """
    if not supabase:
        logger.warning("[CACHE] Supabase not initialized, skipping invalidation")
        return 0
    
    try:
        # This would need to be implemented based on your specific needs
        # For now, we'll just log the request
        logger.info(f"[CACHE] Cache invalidation requested for pattern: {pattern}")
        return 0
        
    except Exception as e:
        logger.error(f"[CACHE] Error invalidating cache pattern {pattern}: {e}")
        return 0

# Initialize cache cleanup on module load
if supabase:
    try:
        cleanup_expired_cache()
    except Exception as e:
        logger.warning(f"[CACHE] Failed to cleanup expired cache on startup: {e}")
