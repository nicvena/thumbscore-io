#!/usr/bin/env python3
"""
Test script for the FAISS index management system
"""

import os
import numpy as np
from app.indices import FAISSIndexManager, rebuild_indices_sync

def test_index_manager():
    """Test the FAISS index manager functionality"""
    
    # Check environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set the following environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_value")
        return
    
    print("✅ Environment variables configured")
    
    try:
        print("🔄 Testing FAISS Index Manager...")
        
        # Initialize manager
        manager = FAISSIndexManager()
        print("✅ Index manager initialized")
        
        # Get initial stats
        stats = manager.get_index_stats()
        print(f"📊 Initial index stats: {stats}")
        
        # Rebuild all indices
        print("🔄 Rebuilding all indices...")
        results = manager.rebuild_all_indices()
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ Index rebuilding completed: {successful}/{total} niches successful")
        
        for niche, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {niche}: {'success' if success else 'failed'}")
        
        # Get updated stats
        stats = manager.get_index_stats()
        print(f"📊 Updated index stats: {stats}")
        
        # Test similarity search (if we have indices)
        for niche in ["tech", "gaming"]:
            if niche in manager.indices and manager.indices[niche].ntotal > 0:
                print(f"🔍 Testing similarity search for {niche}...")
                
                # Create a random query embedding
                query_embedding = np.random.randn(768).astype(np.float32)
                
                # Search for similar thumbnails
                results = manager.search_similar(query_embedding, niche, k=5)
                
                print(f"  Found {len(results)} similar thumbnails")
                for i, result in enumerate(results[:3]):  # Show top 3
                    print(f"    {i+1}. {result['title'][:50]}... (score: {result['similarity_score']:.3f})")
                
                break  # Test one niche only
        
        print("✅ FAISS Index Manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_sync_function():
    """Test the synchronous rebuild function"""
    print("\n🔄 Testing synchronous rebuild function...")
    
    try:
        results = rebuild_indices_sync()
        successful = sum(results.values())
        total = len(results)
        
        print(f"✅ Sync rebuild completed: {successful}/{total} niches successful")
        
        for niche, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {niche}: {'success' if success else 'failed'}")
        
    except Exception as e:
        print(f"❌ Sync test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_index_manager()
    test_sync_function()
