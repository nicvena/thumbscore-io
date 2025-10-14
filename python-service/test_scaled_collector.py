#!/usr/bin/env python3
"""
Test script for the scaled YouTube thumbnail collector
Tests the upgraded collector with smaller limits for verification
"""

import os
import sys
import time
import logging

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_scaled_collector():
    """Test the scaled collector with a small sample"""
    
    print("🚀 Testing Scaled YouTube Thumbnail Collector")
    print("=" * 60)
    
    try:
        # Set environment variables if not already set
        if not os.getenv("YOUTUBE_API_KEY"):
            print("❌ YOUTUBE_API_KEY not set")
            return False
        
        if not os.getenv("SUPABASE_URL"):
            print("❌ SUPABASE_URL not set")
            return False
        
        if not os.getenv("SUPABASE_KEY"):
            print("❌ SUPABASE_KEY not set")
            return False
        
        # Import the collector
        from app.tasks.collect_thumbnails import update_reference_library_sync
        
        # Test with a small limit first (10 videos per niche)
        print("\n🧪 Testing with 10 videos per niche (5 niches = 50 total)")
        print("This will verify the upgraded collector works correctly...")
        
        start_time = time.time()
        stats = update_reference_library_sync(limit_per_niche=10)
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Collection completed in {elapsed_time:.1f}s")
        print("\n📊 RESULTS:")
        print(f"  - Videos fetched: {stats['total_videos_fetched']}")
        print(f"  - Thumbnails stored: {stats['total_thumbnails_stored']}")
        print(f"  - Niches processed: {stats['niches_processed']}")
        print(f"  - Throughput: {stats['throughput_thumbnails_per_sec']:.1f} thumbnails/sec")
        print(f"  - Old thumbnails cleaned: {stats['old_thumbnails_deleted']}")
        
        print("\n📈 NICHE BREAKDOWN:")
        for niche, niche_stats in stats['niche_stats'].items():
            print(f"  - {niche:12s}: {niche_stats['fetched']:2d} fetched | {niche_stats['stored']:2d} stored")
        
        # Verify success
        if stats['total_thumbnails_stored'] > 0:
            print(f"\n🎉 SUCCESS! Collector is working correctly.")
            print(f"   Ready to scale to 200+ videos per niche for 1,000+ total thumbnails")
            return True
        else:
            print(f"\n❌ FAILED! No thumbnails were stored.")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_scale_collection():
    """Test with full scale collection (200 per niche)"""
    
    print("\n" + "=" * 60)
    print("🔥 FULL SCALE TEST - 200 videos per niche")
    print("This will collect ~1,000 thumbnails (5 niches × 200)")
    print("=" * 60)
    
    response = input("\n⚠️  This will take several minutes and use API quota. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Skipping full scale test.")
        return True
    
    try:
        from app.tasks.collect_thumbnails import update_reference_library_sync
        
        print("\n🚀 Starting full scale collection...")
        start_time = time.time()
        
        # Run with default limit (200 per niche)
        stats = update_reference_library_sync()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 FULL SCALE COLLECTION COMPLETED!")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print(f"📊 Total thumbnails collected: {stats['total_thumbnails_stored']}")
        print(f"🚀 Throughput: {stats['throughput_thumbnails_per_sec']:.1f} thumbnails/sec")
        
        # Verify we hit our target
        target_min = 800  # 80% of 1000
        if stats['total_thumbnails_stored'] >= target_min:
            print(f"✅ TARGET ACHIEVED! ({stats['total_thumbnails_stored']} >= {target_min})")
            print("🎯 Your thumbnail library is now at legitimacy-level dataset size!")
            return True
        else:
            print(f"⚠️  Below target ({stats['total_thumbnails_stored']} < {target_min})")
            print("   Check logs for any issues with specific niches")
            return False
        
    except Exception as e:
        print(f"\n❌ Full scale test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("YouTube Thumbnail Collector - Scaled Up Test Suite")
    
    # Test 1: Small scale verification
    success = test_scaled_collector()
    
    if success:
        # Test 2: Full scale collection (optional)
        test_full_scale_collection()
    
    sys.exit(0 if success else 1)
