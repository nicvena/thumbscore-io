#!/usr/bin/env python3
"""
Minimal test to debug FAISS similarity scoring issue
"""

import os
import sys
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_faiss_minimal():
    """Test FAISS similarity scoring in isolation"""
    
    print("🧪 Minimal FAISS Test")
    print("=" * 40)
    
    try:
        # Test 1: Check if cache is loaded
        from app.faiss_cache import is_cache_ready, get_cache_stats, get_index
        
        print("\n1️⃣ Cache Status:")
        cache_ready = is_cache_ready()
        print(f"   Cache Ready: {cache_ready}")
        
        if cache_ready:
            stats = get_cache_stats()
            print(f"   Cached Niches: {stats['cached_niches']}")
            print(f"   Total Items: {stats['total_items']}")
        
        # Test 2: Test index retrieval
        print("\n2️⃣ Index Retrieval:")
        for niche in ['tech', 'business']:
            try:
                result = get_index(niche)
                if result:
                    index, ids = result
                    print(f"   ✅ {niche}: {index.ntotal} vectors")
                else:
                    print(f"   ❌ {niche}: Not found")
            except Exception as e:
                print(f"   ❌ {niche}: Error - {e}")
        
        # Test 3: Test similarity search directly
        print("\n3️⃣ Direct FAISS Search:")
        test_embedding = np.random.randn(768).astype(np.float32)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        for niche in ['tech', 'business']:
            try:
                result = get_index(niche)
                if result:
                    index, ids = result
                    
                    # Prepare query vector
                    query_vec = test_embedding.astype(np.float32)
                    query_vec = query_vec.reshape(1, -1)
                    
                    # Normalize for cosine similarity
                    import faiss
                    faiss.normalize_L2(query_vec)
                    
                    # Search
                    k = min(10, index.ntotal)
                    scores, indices = index.search(query_vec, k)
                    
                    avg_score = float(np.mean(scores[0]))
                    percentile = (avg_score + 1) / 2 * 100
                    
                    print(f"   ✅ {niche}: avg_score={avg_score:.3f}, percentile={percentile:.1f}")
                else:
                    print(f"   ❌ {niche}: No index available")
            except Exception as e:
                print(f"   ❌ {niche}: Search error - {e}")
                import traceback
                traceback.print_exc()
        
        # Test 4: Test get_similarity_score function
        print("\n4️⃣ get_similarity_score Function:")
        from app.ref_library import get_similarity_score
        
        for niche in ['tech', 'business']:
            try:
                score = get_similarity_score(test_embedding, niche)
                print(f"   {niche}: {score}")
            except Exception as e:
                print(f"   ❌ {niche}: Error - {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_faiss_minimal()
