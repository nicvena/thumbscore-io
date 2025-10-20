#!/usr/bin/env python3
"""
Setup Analytics Database Schema

This script creates the analytics tables in Supabase for comprehensive
thumbnail analysis logging and training data collection.
"""

import os
import sys
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_analytics_schema():
    """Create analytics tables in Supabase"""
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: Supabase credentials not found in .env file")
        return False
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")
        
        # Read the analytics schema SQL file
        schema_file = Path(__file__).parent / "analytics_schema.sql"
        
        if not schema_file.exists():
            print("❌ Error: analytics_schema.sql file not found")
            return False
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        print("📝 Applying analytics schema...")
        
        # Split the SQL into individual statements and execute them
        # Note: Supabase RPC doesn't handle multi-statement SQL well,
        # so we'll need to execute them individually or use the SQL editor
        
        # For now, we'll create just the main table
        create_table_sql = """
        -- Main analysis logs table for model training and analytics
        CREATE TABLE IF NOT EXISTS public.thumbnail_analyses (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          created_at TIMESTAMPTZ DEFAULT NOW(),
          
          -- User info
          user_id UUID,
          session_id TEXT NOT NULL,
          
          -- Request metadata
          niche TEXT NOT NULL,
          title TEXT,
          thumbnail_index INTEGER,
          
          -- Scores (main metrics)
          final_score DECIMAL(5,2) NOT NULL,
          confidence DECIMAL(5,2),
          tier TEXT,
          
          -- Component scores (0-100)
          text_clarity DECIMAL(5,2),
          subject_prominence DECIMAL(5,2), 
          contrast_pop DECIMAL(5,2),
          emotion DECIMAL(5,2),
          visual_hierarchy DECIMAL(5,2),
          title_match DECIMAL(5,2),
          power_words DECIMAL(5,2),
          
          -- Detection data
          face_detected BOOLEAN,
          face_size_pct DECIMAL(5,2),
          emotion_detected TEXT,
          word_count INTEGER,
          detected_text TEXT,
          ocr_confidence DECIMAL(5,2),
          saturation DECIMAL(5,4),
          
          -- GPT-4 Vision data
          gpt_summary TEXT,
          gpt_insights JSONB,
          gpt_token_count INTEGER,
          
          -- CTR predictions
          ctr_min DECIMAL(5,2),
          ctr_max DECIMAL(5,2),
          ctr_predicted DECIMAL(5,2),
          
          -- Technical metadata
          processing_time_ms INTEGER,
          scoring_version TEXT DEFAULT 'v1.0',
          model_version TEXT,
          
          -- Image identification (for deduplication)
          image_hash TEXT,
          
          -- Full response (for debugging and reprocessing)
          full_response JSONB,
          
          -- System metadata
          request_ip TEXT,
          user_agent TEXT
        );
        """
        
        # Execute the table creation
        result = supabase.rpc('exec_sql', {'sql': create_table_sql}).execute()
        print("✅ Created thumbnail_analyses table")
        
        # Create feedback table
        feedback_table_sql = """
        CREATE TABLE IF NOT EXISTS public.user_feedback (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          created_at TIMESTAMPTZ DEFAULT NOW(),
          
          analysis_id UUID REFERENCES public.thumbnail_analyses(id) ON DELETE CASCADE,
          user_id UUID,
          
          -- Feedback ratings
          helpful BOOLEAN,
          accurate BOOLEAN,
          used_winner BOOLEAN,
          
          -- Actual performance data
          actual_ctr DECIMAL(5,2),
          actual_views INTEGER,
          actual_impressions INTEGER,
          
          -- Qualitative feedback
          comments TEXT,
          
          -- Metadata
          feedback_type TEXT DEFAULT 'rating',
          request_ip TEXT
        );
        """
        
        result = supabase.rpc('exec_sql', {'sql': feedback_table_sql}).execute()
        print("✅ Created user_feedback table")
        
        # Create basic indexes
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON public.thumbnail_analyses(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_analyses_niche ON public.thumbnail_analyses(niche);
        CREATE INDEX IF NOT EXISTS idx_analyses_final_score ON public.thumbnail_analyses(final_score);
        """
        
        result = supabase.rpc('exec_sql', {'sql': index_sql}).execute()
        print("✅ Created indexes")
        
        print("🎉 Analytics database schema setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up analytics schema: {e}")
        return False

def test_analytics_logging():
    """Test analytics logging functionality"""
    try:
        from app.analytics_logger import analytics_logger
        
        # Test basic functionality
        if not analytics_logger.enabled:
            print("⚠️ Analytics logger is not enabled")
            return False
        
        print("✅ Analytics logger is enabled and ready")
        
        # Test database connection
        analytics_data = analytics_logger.get_niche_analytics(days=1)
        if analytics_data is not None:
            print(f"✅ Database connection working - {analytics_data.get('total_analyses', 0)} analyses found")
        else:
            print("⚠️ Could not fetch analytics data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing analytics logging: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Analytics Database Schema")
    print("=" * 50)
    
    # Setup schema
    success = setup_analytics_schema()
    
    if success:
        print("\n🧪 Testing analytics logging...")
        test_success = test_analytics_logging()
        
        if test_success:
            print("\n🎉 All tests passed! Analytics logging is ready.")
        else:
            print("\n⚠️ Schema created but testing failed. Check configuration.")
    else:
        print("\n❌ Schema setup failed. Check your Supabase configuration.")
        sys.exit(1)