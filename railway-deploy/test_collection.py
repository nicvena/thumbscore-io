#!/usr/bin/env python3
"""
Test script for the automated thumbnail collection system
"""

import os
import asyncio
from app.tasks.collect_thumbnails import update_reference_library

async def test_collection():
    """Test the thumbnail collection system"""
    
    # Check environment variables
    required_vars = ["YOUTUBE_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set the following environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_value")
        return
    
    print("✅ Environment variables configured")
    
    try:
        print("🔄 Starting thumbnail collection test...")
        stats = await update_reference_library()
        
        print("✅ Collection completed successfully!")
        print(f"📊 Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_collection())
