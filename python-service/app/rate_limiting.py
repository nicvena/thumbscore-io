#!/usr/bin/env python3
"""
Rate Limiting System for Thumbscore.io

Prevents abuse and controls API usage with IP-based rate limiting.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
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
    logger.error("[RATE_LIMIT] Missing Supabase credentials")
    supabase: Optional[Client] = None
else:
    supabase: Client = create_client(supabase_url, supabase_key)

# Rate limit configurations
RATE_LIMITS = {
    "preview": {
        "requests_per_hour": 1000,  # Increased for testing
        "window_minutes": 60,
        "endpoint": "/v1/preview_score"
    },
    "full_analysis": {
        "requests_per_hour": 100,   # Increased for testing
        "window_minutes": 60,
        "endpoint": "/v1/score"
    }
}

def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request
    Handles various proxy headers
    """
    # Check for forwarded IP (common in production)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Check for CloudFlare IP
    cf_connecting_ip = request.headers.get("CF-Connecting-IP")
    if cf_connecting_ip:
        return cf_connecting_ip
    
    # Fallback to direct client IP
    if hasattr(request, 'client') and request.client:
        return request.client.host
    
    # Ultimate fallback
    return "127.0.0.1"

def check_rate_limit(request: Request, limit_type: str) -> Dict[str, Any]:
    """
    Check if request is within rate limits
    Returns rate limit status
    """
    if not supabase:
        logger.warning("[RATE_LIMIT] Supabase not initialized, allowing request")
        return {"allowed": True, "remaining": 999, "reset_time": None}
    
    if limit_type not in RATE_LIMITS:
        logger.error(f"[RATE_LIMIT] Unknown limit type: {limit_type}")
        return {"allowed": True, "remaining": 999, "reset_time": None}
    
    try:
        # Get client IP
        client_ip = get_client_ip(request)
        
        # Get rate limit config
        config = RATE_LIMITS[limit_type]
        max_requests = config["requests_per_hour"]
        window_minutes = config["window_minutes"]
        
        # Calculate window start time
        now = datetime.now()
        window_start = now.replace(minute=(now.minute // window_minutes) * window_minutes, second=0, microsecond=0)
        
        # Check current hits for this IP and window
        result = supabase.table("rate_limits").select("hits").eq("ip", client_ip).eq("window_start", window_start.isoformat()).execute()
        
        if result.data:
            current_hits = result.data[0]["hits"]
        else:
            current_hits = 0
        
        # Check if limit exceeded
        if current_hits >= max_requests:
            logger.warning(f"[RATE_LIMIT] Rate limit exceeded for IP {client_ip}: {current_hits}/{max_requests}")
            
            # Calculate reset time
            reset_time = window_start + timedelta(minutes=window_minutes)
            
            return {
                "allowed": False,
                "remaining": 0,
                "reset_time": reset_time.isoformat(),
                "limit": max_requests,
                "window_minutes": window_minutes,
                "ip": client_ip
            }
        
        # Increment hit counter
        if current_hits == 0:
            # Insert new record
            supabase.table("rate_limits").insert({
                "ip": client_ip,
                "window_start": window_start.isoformat(),
                "hits": 1
            }).execute()
        else:
            # Update existing record
            supabase.table("rate_limits").update({
                "hits": current_hits + 1
            }).eq("ip", client_ip).eq("window_start", window_start.isoformat()).execute()
        
        remaining = max_requests - (current_hits + 1)
        
        logger.info(f"[RATE_LIMIT] Request allowed for IP {client_ip}: {current_hits + 1}/{max_requests} (remaining: {remaining})")
        
        return {
            "allowed": True,
            "remaining": remaining,
            "reset_time": (window_start + timedelta(minutes=window_minutes)).isoformat(),
            "limit": max_requests,
            "window_minutes": window_minutes,
            "ip": client_ip
        }
        
    except Exception as e:
        logger.error(f"[RATE_LIMIT] Error checking rate limit: {e}")
        # Allow request on error to avoid blocking legitimate users
        return {"allowed": True, "remaining": 999, "reset_time": None, "error": str(e)}

def cleanup_rate_limits() -> int:
    """
    Clean up old rate limit entries
    """
    if not supabase:
        logger.warning("[RATE_LIMIT] Supabase not initialized, skipping cleanup")
        return 0
    
    try:
        # Call the SQL function to cleanup old entries
        result = supabase.rpc("cleanup_rate_limits").execute()
        
        logger.info("[RATE_LIMIT] Cleaned up old rate limit entries")
        return 0  # Supabase doesn't return count for cleanup functions
        
    except Exception as e:
        logger.error(f"[RATE_LIMIT] Error cleaning up rate limits: {e}")
        return 0

def get_rate_limit_stats() -> Dict[str, Any]:
    """
    Get rate limiting statistics
    """
    if not supabase:
        return {"error": "Supabase not initialized"}
    
    try:
        # Get total rate limit entries
        total_result = supabase.table("rate_limits").select("ip", count="exact").execute()
        total_entries = total_result.count if hasattr(total_result, 'count') else len(total_result.data)
        
        # Get recent entries (last hour)
        recent_cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
        recent_result = supabase.table("rate_limits").select("ip", count="exact").gte("window_start", recent_cutoff).execute()
        recent_entries = recent_result.count if hasattr(recent_result, 'count') else len(recent_result.data)
        
        return {
            "total_entries": total_entries,
            "recent_entries_1h": recent_entries,
            "limits": RATE_LIMITS
        }
        
    except Exception as e:
        logger.error(f"[RATE_LIMIT] Error getting rate limit stats: {e}")
        return {"error": str(e)}

# Initialize cleanup on module load
if supabase:
    try:
        cleanup_rate_limits()
    except Exception as e:
        logger.warning(f"[RATE_LIMIT] Failed to cleanup rate limits on startup: {e}")
