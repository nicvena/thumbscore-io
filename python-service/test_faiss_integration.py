#!/usr/bin/env python3
"""
Comprehensive test script for FAISS integration in Thumbnail Lab

Tests the complete flow:
1. Build FAISS indices from Supabase
2. Load indices into memory
3. Perform similarity searches
4. Test ref_library integration
"""

import os
import sys
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_environment():
    """Test that all required environment variables are set"""
    logger.info("Testing environment variables...")
    
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        logger.info("Please set the following environment variables:")
        for var in missing_vars:
            logger.info(f"  export {var}=your_value")
        return False
    
    logger.info("✅ Environment variables configured")
    return True


def test_build_indices():
    """Test building FAISS indices"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Building FAISS Indices")
    logger.info("="*60)
    
    try:
        from app.tasks.build_faiss_index import build_faiss_indices, get_faiss_index_info
        
        # Build indices
        logger.info("Building FAISS indices...")
        results = build_faiss_indices()
        
        successful = sum(results.values())
        total = len(results)
        
        logger.info(f"\nResults: {successful}/{total} niches successful")
        for niche, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {niche}")
        
        # Get index info
        logger.info("\nIndex Information:")
        info = get_faiss_index_info()
        for niche, details in info.items():
            if details.get("exists"):
                num_vectors = details.get("num_vectors", 0)
                file_size = details.get("file_size_mb", 0)
                logger.info(f"  {niche}: {num_vectors} vectors ({file_size:.2f} MB)")
            else:
                logger.info(f"  {niche}: not found")
        
        return successful > 0
        
    except Exception as e:
        logger.error(f"❌ Build indices test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_indices():
    """Test loading FAISS indices"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Loading FAISS Indices")
    logger.info("="*60)
    
    try:
        from app.ref_library import load_faiss_index
        
        test_niches = ["tech", "gaming", "education"]
        loaded_count = 0
        
        for niche in test_niches:
            logger.info(f"\nLoading index for {niche}...")
            result = load_faiss_index(niche)
            
            if result:
                index, video_ids = result
                logger.info(f"✅ {niche}: {index.ntotal} vectors, {len(video_ids)} IDs")
                loaded_count += 1
            else:
                logger.warning(f"❌ {niche}: not found")
        
        logger.info(f"\nLoaded {loaded_count}/{len(test_niches)} indices")
        return loaded_count > 0
        
    except Exception as e:
        logger.error(f"❌ Load indices test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_similarity_search():
    """Test similarity search with random vectors"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Similarity Search")
    logger.info("="*60)
    
    try:
        from app.ref_library import (
            faiss_similarity_percentile,
            faiss_get_similar_thumbnails,
            get_similarity_score
        )
        
        # Create random test embedding
        test_embedding = np.random.randn(768).astype(np.float32)
        
        test_niches = ["tech", "gaming"]
        
        for niche in test_niches:
            logger.info(f"\nTesting {niche}...")
            
            # Test percentile calculation
            percentile = faiss_similarity_percentile(test_embedding, niche, top_k=100)
            if percentile is not None:
                logger.info(f"  Similarity percentile: {percentile:.2f}")
            else:
                logger.warning(f"  Percentile calculation failed")
            
            # Test get_similarity_score (with fallback)
            score = get_similarity_score(test_embedding, niche)
            logger.info(f"  Overall similarity score: {score:.2f}")
            
            # Test getting similar thumbnails
            similar = faiss_get_similar_thumbnails(test_embedding, niche, top_k=5)
            logger.info(f"  Found {len(similar)} similar thumbnails")
            
            if similar:
                logger.info("  Top 3 similar:")
                for i, item in enumerate(similar[:3]):
                    video_id = item.get("video_id", "unknown")
                    sim_score = item.get("similarity_score", 0)
                    logger.info(f"    {i+1}. {video_id} (score: {sim_score:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Similarity search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_functionality():
    """Test FAISS index caching"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Cache Functionality")
    logger.info("="*60)
    
    try:
        from app.ref_library import (
            load_faiss_index,
            clear_index_cache,
            get_index_cache_stats
        )
        
        # Load some indices
        logger.info("Loading indices...")
        load_faiss_index("tech")
        load_faiss_index("gaming")
        
        # Check cache stats
        stats = get_index_cache_stats()
        logger.info(f"Cache stats: {stats}")
        logger.info(f"✅ Cached niches: {stats.get('cached_niches', 0)}")
        logger.info(f"  Niches: {stats.get('niches', [])}")
        
        # Clear cache
        logger.info("\nClearing cache...")
        clear_index_cache()
        
        # Check cache stats again
        stats = get_index_cache_stats()
        logger.info(f"Cache stats after clear: {stats}")
        logger.info(f"✅ Cached niches: {stats.get('cached_niches', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_integration():
    """Test that indices exist in the expected location"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: File System Integration")
    logger.info("="*60)
    
    try:
        import os
        
        faiss_path = os.getenv("FAISS_INDEX_PATH", "faiss_indices")
        
        if not os.path.exists(faiss_path):
            logger.warning(f"❌ FAISS index directory not found: {faiss_path}")
            return False
        
        logger.info(f"✅ FAISS index directory exists: {faiss_path}")
        
        # List files
        files = os.listdir(faiss_path)
        index_files = [f for f in files if f.endswith('.index')]
        metadata_files = [f for f in files if f.endswith('_ids.npy')]
        
        logger.info(f"\nFound {len(index_files)} index files:")
        for f in index_files:
            size_mb = os.path.getsize(os.path.join(faiss_path, f)) / (1024 * 1024)
            logger.info(f"  {f} ({size_mb:.2f} MB)")
        
        logger.info(f"\nFound {len(metadata_files)} metadata files:")
        for f in metadata_files:
            size_kb = os.path.getsize(os.path.join(faiss_path, f)) / 1024
            logger.info(f"  {f} ({size_kb:.2f} KB)")
        
        return len(index_files) > 0
        
    except Exception as e:
        logger.error(f"❌ File system integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("FAISS Integration Test Suite")
    logger.info("="*60)
    
    # Test environment
    if not test_environment():
        logger.error("Environment setup failed. Exiting.")
        return 1
    
    # Run tests
    results = {
        "Build Indices": test_build_indices(),
        "Load Indices": test_load_indices(),
        "Similarity Search": test_similarity_search(),
        "Cache Functionality": test_cache_functionality(),
        "File System Integration": test_api_integration()
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

