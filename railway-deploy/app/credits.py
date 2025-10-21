#!/usr/bin/env python3
"""
Credit Management System for Thumbscore.io

Handles user authentication, credit tracking, and cost control
to prevent unexpected OpenAI Vision API charges.
"""

import os
import uuid
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, date
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
    logger.error("[CREDITS] Missing Supabase credentials")
    supabase: Optional[Client] = None
else:
    supabase: Client = create_client(supabase_url, supabase_key)

# Plan configurations
PLAN_CONFIGS = {
    "free": {
        "monthly_quota": 50,  # Increased for testing
        "max_thumbnails_per_analysis": 3,
        "name": "Free",
        "description": "1 AI analysis per month (max 3 thumbnails)"
    },
    "creator": {
        "monthly_quota": 50,
        "max_thumbnails_per_analysis": 10,
        "name": "Creator",
        "description": "50 AI analyses per month (max 10 thumbnails each)"
    },
    "pro": {
        "monthly_quota": 200,
        "max_thumbnails_per_analysis": 20,
        "name": "Pro",
        "description": "200 AI analyses per month (max 20 thumbnails each)"
    }
}

def get_user_from_request(request: Request) -> Tuple[str, str]:
    """
    Extract user_id and plan from request headers or cookies
    Supports both authenticated users (X-User-Id header) and anonymous users (device_id cookie)
    
    Returns:
        Tuple[user_id, plan]
    """
    try:
        # Check for authenticated user ID in header
        user_id = request.headers.get("X-User-Id")
        if user_id:
            logger.info(f"[CREDITS] Authenticated user: {user_id}")
            return user_id, "authenticated"
        
        # Check for anonymous device ID in cookie
        device_id = request.cookies.get("device_id")
        if device_id:
            logger.info(f"[CREDITS] Anonymous user: {device_id}")
            return device_id, "anonymous"
        
        # Generate new anonymous device ID
        device_id = str(uuid.uuid4())
        logger.info(f"[CREDITS] New anonymous user: {device_id}")
        return device_id, "anonymous"
        
    except Exception as e:
        logger.error(f"[CREDITS] Error getting user from request: {e}")
        # Fallback to anonymous
        device_id = str(uuid.uuid4())
        return device_id, "anonymous"

def ensure_wallet(user_id: str) -> Dict[str, Any]:
    """
    Ensure user has a credit wallet, create if missing
    Returns wallet information
    """
    if not supabase:
        logger.error("[CREDITS] Supabase not initialized")
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Check if user exists in users table
        user_result = supabase.table("users").select("id").eq("id", user_id).execute()
        
        if not user_result.data:
            # Create user record
            user_data = {
                "id": user_id,
                "device_id": user_id if len(user_id) > 20 else None,  # Assume device_id if long UUID
                "email": None
            }
            supabase.table("users").insert(user_data).execute()
            logger.info(f"[CREDITS] Created user record: {user_id}")
        
        # Check if credits record exists
        credits_result = supabase.table("credits").select("*").eq("user_id", user_id).execute()
        
        if not credits_result.data:
            # Create credits record with free plan
            credits_data = {
                "user_id": user_id,
                "plan": "free",
                "monthly_quota": PLAN_CONFIGS["free"]["monthly_quota"],
                "max_thumbnails_per_analysis": PLAN_CONFIGS["free"]["max_thumbnails_per_analysis"],
                "used_this_cycle": 0,
                "cycle_start": date.today().replace(day=1).isoformat()  # First day of current month
            }
            supabase.table("credits").insert(credits_data).execute()
            logger.info(f"[CREDITS] Created credits wallet for user: {user_id}")
            
            return credits_data
        else:
            wallet = credits_result.data[0]
            logger.info(f"[CREDITS] Found existing wallet for user: {user_id}, plan: {wallet['plan']}")
            return wallet
            
    except Exception as e:
        logger.error(f"[CREDITS] Error ensuring wallet for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ensure wallet: {str(e)}")

def reset_monthly_credits() -> None:
    """
    Reset monthly credit cycles for all users
    Should be called periodically (e.g., daily cron job)
    """
    if not supabase:
        logger.error("[CREDITS] Supabase not initialized")
        return
    
    try:
        # Call the SQL function to reset credits
        result = supabase.rpc("reset_credits_monthly", {}).execute()
        logger.info("[CREDITS] Monthly credit reset completed")
        
    except Exception as e:
        logger.error(f"[CREDITS] Error resetting monthly credits: {e}")

def check_and_consume_credit(user_id: str) -> Dict[str, Any]:
    """
    Check if user can consume a credit and consume it if possible
    Returns credit status and updated wallet info
    
    Raises HTTPException(402) if no credits available
    """
    if not supabase:
        logger.error("[CREDITS] Supabase not initialized")
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Ensure wallet exists
        wallet = ensure_wallet(user_id)
        
        # Check if user has credits available
        plan = wallet["plan"]
        monthly_quota = wallet["monthly_quota"]
        used_this_cycle = wallet["used_this_cycle"]
        
        if used_this_cycle >= monthly_quota:
            logger.warning(f"[CREDITS] User {user_id} exceeded quota: {used_this_cycle}/{monthly_quota}")
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "quota_exceeded",
                    "message": f"You've used all {monthly_quota} credits for this month",
                    "plan": plan,
                    "used": used_this_cycle,
                    "quota": monthly_quota,
                    "upgrade_url": "/#pricing"
                }
            )
        
        # Consume credit
        new_used = used_this_cycle + 1
        supabase.table("credits").update({
            "used_this_cycle": new_used,
            "updated_at": datetime.now().isoformat()
        }).eq("user_id", user_id).execute()
        
        remaining = monthly_quota - new_used
        
        logger.info(f"[CREDITS] User {user_id} consumed credit: {new_used}/{monthly_quota} (remaining: {remaining})")
        
        return {
            "success": True,
            "plan": plan,
            "used": new_used,
            "quota": monthly_quota,
            "remaining": remaining,
            "wallet": wallet
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CREDITS] Error checking/consuming credit for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check credits: {str(e)}")

def check_thumbnail_limit(user_id: str, thumbnail_count: int) -> Dict[str, Any]:
    """
    Check if user can analyze the requested number of thumbnails
    Returns status and limit information
    """
    if not supabase:
        logger.warning("[CREDITS] Supabase not initialized, using fallback limits")
        # Fallback to free plan limits when Supabase is not available
        max_thumbnails = PLAN_CONFIGS["free"]["max_thumbnails_per_analysis"]
        plan = "free"
        
        if thumbnail_count > max_thumbnails:
            logger.warning(f"[CREDITS] User {user_id} exceeded thumbnail limit: {thumbnail_count}/{max_thumbnails}")
            return {
                "allowed": False,
                "requested": thumbnail_count,
                "max_allowed": max_thumbnails,
                "plan": plan,
                "upgrade_required": True,
                "message": f"Free plan allows max {max_thumbnails} thumbnails per analysis. Upgrade to analyze more."
            }
        
        return {
            "allowed": True,
            "requested": thumbnail_count,
            "max_allowed": max_thumbnails,
            "plan": plan
        }
    
    try:
        wallet = ensure_wallet(user_id)
        plan = wallet["plan"]
        
        max_thumbnails = PLAN_CONFIGS.get(plan, {}).get("max_thumbnails_per_analysis", 3)
        
        if thumbnail_count > max_thumbnails:
            logger.warning(f"[CREDITS] User {user_id} exceeded thumbnail limit: {thumbnail_count}/{max_thumbnails}")
            return {
                "allowed": False,
                "requested": thumbnail_count,
                "max_allowed": max_thumbnails,
                "plan": plan,
                "upgrade_required": True,
                "message": f"Free plan allows max {max_thumbnails} thumbnails per analysis. Upgrade to analyze more."
            }
        
        return {
            "allowed": True,
            "requested": thumbnail_count,
            "max_allowed": max_thumbnails,
            "plan": plan
        }
        
    except Exception as e:
        logger.error(f"[CREDITS] Error checking thumbnail limit for user {user_id}: {e}")
        # Fallback to free plan limits on error
        max_thumbnails = PLAN_CONFIGS["free"]["max_thumbnails_per_analysis"]
        plan = "free"
        
        if thumbnail_count > max_thumbnails:
            return {
                "allowed": False,
                "requested": thumbnail_count,
                "max_allowed": max_thumbnails,
                "plan": plan,
                "upgrade_required": True,
                "message": f"Free plan allows max {max_thumbnails} thumbnails per analysis. Upgrade to analyze more."
            }
        
        return {
            "allowed": True,
            "requested": thumbnail_count,
            "max_allowed": max_thumbnails,
            "plan": plan
        }

def get_credit_status(user_id: str) -> Dict[str, Any]:
    """
    Get current credit status for user (without consuming)
    """
    if not supabase:
        logger.error("[CREDITS] Supabase not initialized")
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        wallet = ensure_wallet(user_id)
        
        plan = wallet["plan"]
        monthly_quota = wallet["monthly_quota"]
        used_this_cycle = wallet["used_this_cycle"]
        remaining = monthly_quota - used_this_cycle
        max_thumbnails = PLAN_CONFIGS.get(plan, {}).get("max_thumbnails_per_analysis", 3)
        
        return {
            "plan": plan,
            "plan_name": PLAN_CONFIGS.get(plan, {}).get("name", "Unknown"),
            "plan_description": PLAN_CONFIGS.get(plan, {}).get("description", ""),
            "used": used_this_cycle,
            "quota": monthly_quota,
            "remaining": remaining,
            "cycle_start": wallet["cycle_start"],
            "can_analyze": remaining > 0,
            "max_thumbnails_per_analysis": max_thumbnails
        }
        
    except Exception as e:
        logger.error(f"[CREDITS] Error getting credit status for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get credit status: {str(e)}")

def upgrade_user_plan(user_id: str, new_plan: str) -> Dict[str, Any]:
    """
    Upgrade user to a new plan
    """
    if not supabase:
        logger.error("[CREDITS] Supabase not initialized")
        raise HTTPException(status_code=500, detail="Database not available")
    
    if new_plan not in PLAN_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {new_plan}")
    
    try:
        # Update user's plan and quota
        new_quota = PLAN_CONFIGS[new_plan]["monthly_quota"]
        new_thumbnail_limit = PLAN_CONFIGS[new_plan]["max_thumbnails_per_analysis"]
        
        supabase.table("credits").update({
            "plan": new_plan,
            "monthly_quota": new_quota,
            "max_thumbnails_per_analysis": new_thumbnail_limit,
            "used_this_cycle": 0,  # Reset usage for new plan
            "cycle_start": date.today().replace(day=1).isoformat(),
            "updated_at": datetime.now().isoformat()
        }).eq("user_id", user_id).execute()
        
        logger.info(f"[CREDITS] Upgraded user {user_id} to plan: {new_plan} (quota: {new_quota})")
        
        return {
            "success": True,
            "plan": new_plan,
            "quota": new_quota,
            "message": f"Successfully upgraded to {PLAN_CONFIGS[new_plan]['name']}"
        }
        
    except Exception as e:
        logger.error(f"[CREDITS] Error upgrading user {user_id} to plan {new_plan}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upgrade plan: {str(e)}")

# Initialize monthly reset on module load
if supabase:
    try:
        reset_monthly_credits()
    except Exception as e:
        logger.warning(f"[CREDITS] Failed to reset monthly credits on startup: {e}")
