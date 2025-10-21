#!/usr/bin/env python3
"""
Test script for FAISS cache system
"""

import os
import sys
import time
import logging

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_faiss_cache():
    """Test the FAISS cache system"""
    
    print("🧪 Testing FAISS Cache System")
    print("=" * 50)
    
    try:
        # Import cache functions
        from app.faiss_cache import (
            load_indices, 
            get_cache_stats, 
            get_index, 
            is_cache_ready,
            get_available_niches
        )
        
        # Test 1: Check if indices exist
        print("\n1️⃣ Checking for existing FAISS indices...")
        faiss_dir = "faiss_indices"
        if os.path.exists(faiss_dir):
            index_files = [f for f in os.listdir(faiss_dir) if f.endswith('.index')]
            print(f"   Found {len(index_files)} index files: {index_files}")
        else:
            print("   ❌ No faiss_indices directory found")
            return False
        
        # Test 2: Load indices into cache
        print("\n2️⃣ Loading indices into cache...")
        start_time = time.time()
        load_indices()
        load_time = time.time() - start_time
        
        if is_cache_ready():
            print(f"   ✅ Cache loaded successfully in {load_time:.3f}s")
        else:
            print("   ❌ Cache failed to load")
            return False
        
        # Test 3: Get cache statistics
        print("\n3️⃣ Cache Statistics:")
        stats = get_cache_stats()
        print(f"   - Cached niches: {stats['total_niches']}")
        print(f"   - Total items: {stats['total_items']}")
        print(f"   - Memory usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"   - Available niches: {stats['cached_niches']}")
        
        # Test 4: Test index retrieval
        print("\n4️⃣ Testing index retrieval...")
        available_niches = get_available_niches()
        
        if available_niches:
            test_niche = available_niches[0]
            print(f"   Testing retrieval for niche: {test_niche}")
            
            start_time = time.time()
            result = get_index(test_niche)
            retrieval_time = time.time() - start_time
            
            if result:
                index, ids = result
                print(f"   ✅ Retrieved index with {index.ntotal} vectors in {retrieval_time*1000:.2f}ms")
                print(f"   - Index type: {type(index).__name__}")
                print(f"   - IDs shape: {ids.shape if len(ids) > 0 else 'empty'}")
            else:
                print("   ❌ Failed to retrieve index")
                return False
        else:
            print("   ⚠️  No niches available for testing")
        
        # Test 5: Performance test
        print("\n5️⃣ Performance Test:")
        if available_niches:
            test_niche = available_niches[0]
            iterations = 100
            
            start_time = time.time()
            for _ in range(iterations):
                get_index(test_niche)
            total_time = time.time() - start_time
            avg_time = total_time / iterations * 1000
            
            print(f"   - {iterations} retrievals in {total_time:.3f}s")
            print(f"   - Average: {avg_time:.2f}ms per retrieval")
            print(f"   - Performance: {'🚀 EXCELLENT' if avg_time < 1 else '✅ GOOD' if avg_time < 10 else '⚠️ SLOW'}")
        
        print("\n✅ FAISS Cache System Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAISS Cache System Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_faiss_cache()
    sys.exit(0 if success else 1)
