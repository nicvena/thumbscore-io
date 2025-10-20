#!/usr/bin/env python3
"""
Test Analytics Logging

Simple test to verify analytics logging is working correctly.
"""

import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_supabase_connection():
    """Test basic Supabase connection"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not found")
        return False
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Test connection by fetching existing tables
        result = supabase.table('users').select('count', count='exact').execute()
        print(f"✅ Supabase connection works - {result.count} users in database")
        
        # Check if thumbnail_analyses table exists
        try:
            result = supabase.table('thumbnail_analyses').select('count', count='exact').execute()
            print(f"✅ thumbnail_analyses table exists with {result.count} records")
            return True
        except Exception as e:
            print(f"⚠️ thumbnail_analyses table doesn't exist: {e}")
            print("📝 Creating table manually...")
            
            # Try to create the table using SQL
            create_sql = """
            CREATE TABLE IF NOT EXISTS public.thumbnail_analyses (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              created_at TIMESTAMPTZ DEFAULT NOW(),
              user_id UUID,
              session_id TEXT NOT NULL,
              niche TEXT NOT NULL,
              title TEXT,
              thumbnail_index INTEGER,
              final_score DECIMAL(5,2) NOT NULL,
              confidence DECIMAL(5,2),
              tier TEXT,
              text_clarity DECIMAL(5,2),
              subject_prominence DECIMAL(5,2),
              contrast_pop DECIMAL(5,2),
              emotion DECIMAL(5,2),
              visual_hierarchy DECIMAL(5,2),
              title_match DECIMAL(5,2),
              power_words DECIMAL(5,2),
              face_detected BOOLEAN,
              face_size_pct DECIMAL(5,2),
              emotion_detected TEXT,
              word_count INTEGER,
              detected_text TEXT,
              ocr_confidence DECIMAL(5,2),
              saturation DECIMAL(5,4),
              gpt_summary TEXT,
              gpt_insights JSONB,
              gpt_token_count INTEGER,
              ctr_min DECIMAL(5,2),
              ctr_max DECIMAL(5,2),
              ctr_predicted DECIMAL(5,2),
              processing_time_ms INTEGER,
              scoring_version TEXT DEFAULT 'v1.0',
              model_version TEXT,
              image_hash TEXT,
              full_response JSONB,
              request_ip TEXT,
              user_agent TEXT
            );
            """
            
            # Use the SQL editor endpoint
            print("ℹ️ You need to run this SQL in the Supabase SQL editor:")
            print("=" * 60)
            print(create_sql)
            print("=" * 60)
            
            return False
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False

def test_analytics_logger():
    """Test the analytics logger"""
    try:
        from app.analytics_logger import analytics_logger
        
        print(f"✅ Analytics logger enabled: {analytics_logger.enabled}")
        
        if analytics_logger.enabled:
            # Test analytics fetching
            analytics = analytics_logger.get_niche_analytics(days=7)
            print(f"✅ Analytics fetch works: {analytics}")
            
        return True
        
    except Exception as e:
        print(f"❌ Analytics logger test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Analytics Setup")
    print("=" * 40)
    
    # Test Supabase connection
    conn_success = test_supabase_connection()
    
    # Test analytics logger
    logger_success = test_analytics_logger()
    
    if conn_success and logger_success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check configuration.")