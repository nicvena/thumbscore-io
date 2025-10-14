"""
Reference Library Similarity Scoring

This module provides FAISS-based similarity lookups for thumbnail scoring.
It uses the centralized FAISS cache for instant similarity searches.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np
import faiss
from supabase import create_client, Client

# Import the centralized FAISS cache
from app.faiss_cache import get_index, get_cache_stats, is_cache_ready

logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_indices")
EMBEDDING_DIM = 768


def load_faiss_index(niche: str) -> Optional[Tuple[faiss.Index, np.ndarray]]:
    """
    Get FAISS index and reference IDs for a specific niche from cache.
    Uses centralized cache for instant access.
    
    Args:
        niche: Niche category name
        
    Returns:
        Tuple of (faiss_index, video_ids_array) or None if not found
    """
    # Use centralized cache
    result = get_index(niche)
    if result is None:
        logger.warning(f"FAISS index for {niche} not found in cache")
        return None
    
    index, video_ids = result
    logger.debug(f"Retrieved cached FAISS index for {niche} with {index.ntotal} vectors")
    return index, video_ids


def clear_index_cache():
    """Clear the global FAISS index cache. Useful after rebuilding indices."""
    from app.faiss_cache import clear_cache
    clear_cache()


def faiss_similarity_percentile(
    upload_vec: np.ndarray, 
    niche: str, 
    top_k: int = 200
) -> Optional[float]:
    """
    Calculate similarity percentile score using FAISS index.
    
    Now supports deterministic search for consistent results.
    
    Args:
        upload_vec: User's thumbnail embedding (768-dim)
        niche: Niche category to search in
        top_k: Number of similar results to retrieve
        
    Returns:
        Percentile score (0-100) or None if index not available
    """
    try:
        # Load FAISS index
        result = load_faiss_index(niche)
        if result is None:
            logger.warning(f"FAISS index not available for {niche}")
            return None
        
        index, video_ids = result
        
        if index.ntotal == 0:
            logger.warning(f"Empty FAISS index for {niche}")
            return None
        
        # Prepare query vector with deterministic rounding
        query_vec = upload_vec.astype(np.float32)
        query_vec = np.round(query_vec, decimals=4)  # Deterministic rounding
        query_vec = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vec)  # Normalize for cosine similarity
        
        # Use deterministic search if available
        try:
            from app.determinism import deterministic_faiss_search
            scores, indices = deterministic_faiss_search(index, query_vec[0], top_k)
        except ImportError:
            # Fallback to regular search
            k = min(top_k, index.ntotal)
            scores, indices = index.search(query_vec, k)
            scores = scores[0]
            indices = indices[0]
        
        # Calculate percentile with deterministic averaging
        avg_similarity = float(np.mean(scores))
        
        # Convert to percentile (0-100 scale)
        # Similarity scores typically range from -1 to 1 (after normalization)
        # We map this to 0-100 percentile
        percentile = (avg_similarity + 1) / 2 * 100
        percentile = max(0, min(100, percentile))  # Clamp to 0-100
        
        logger.debug(f"Similarity percentile for {niche}: {percentile:.2f}")
        return percentile
        
    except Exception as e:
        logger.error(f"Error calculating FAISS similarity for {niche}: {e}")
        return None


def faiss_get_similar_thumbnails(
    upload_vec: np.ndarray,
    niche: str,
    top_k: int = 10
) -> List[Dict]:
    """
    Get top-k similar thumbnails from the reference library using FAISS.
    
    Args:
        upload_vec: User's thumbnail embedding (768-dim)
        niche: Niche category to search in
        top_k: Number of similar results to retrieve
        
    Returns:
        List of dictionaries with video_id and similarity_score
    """
    try:
        # Load FAISS index
        result = load_faiss_index(niche)
        if result is None:
            return []
        
        index, video_ids = result
        
        if index.ntotal == 0:
            return []
        
        # Prepare query vector
        query_vec = upload_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)
        
        # Search for top-k similar thumbnails
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vec, k)
        
        # Build results list
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(video_ids):
                results.append({
                    "video_id": str(video_ids[idx]),
                    "similarity_score": float(score)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting similar thumbnails for {niche}: {e}")
        return []


def fallback_supabase_similarity(
    upload_vec: np.ndarray,
    niche: str,
    top_k: int = 200
) -> Optional[float]:
    """
    Fallback similarity calculation using Supabase vector search.
    Used when FAISS index is not available.
    
    Args:
        upload_vec: User's thumbnail embedding (768-dim)
        niche: Niche category to search in
        top_k: Number of similar results to retrieve
        
    Returns:
        Percentile score (0-100) or None if failed
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase credentials not configured")
        return None
    
    try:
        logger.info(f"Using Supabase fallback for {niche} similarity")
        
        # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Convert embedding to list for Supabase
        embedding_list = upload_vec.tolist()
        
        # Query Supabase for similar thumbnails
        # Note: This requires pgvector extension and proper setup
        result = supabase.rpc(
            "match_thumbnails",
            {
                "query_embedding": embedding_list,
                "match_niche": niche,
                "match_count": top_k
            }
        ).execute()
        
        if not result.data or len(result.data) == 0:
            logger.warning(f"No results from Supabase similarity for {niche}")
            return 50.0  # Default middle percentile
        
        # Calculate average similarity from results
        similarities = [row.get("similarity", 0) for row in result.data]
        avg_similarity = np.mean(similarities)
        
        # Convert to percentile
        percentile = (avg_similarity + 1) / 2 * 100
        percentile = max(0, min(100, percentile))
        
        return percentile
        
    except Exception as e:
        logger.error(f"Supabase fallback failed for {niche}: {e}")
        return None  # Let caller handle the fallback


def get_similarity_score(
    upload_vec: np.ndarray,
    niche: str,
    top_k: int = 200
) -> float:
    """
    Get similarity score for a thumbnail embedding.
    Tries FAISS first, falls back to Supabase if needed.
    
    Args:
        upload_vec: User's thumbnail embedding (768-dim)
        niche: Niche category to search in
        top_k: Number of similar results to retrieve
        
    Returns:
        Percentile score (0-100)
    """
    # Try FAISS first
    score = faiss_similarity_percentile(upload_vec, niche, top_k)
    
    # Fallback to Supabase if FAISS not available
    if score is None:
        logger.info(f"FAISS not available for {niche}, using Supabase fallback")
        score = fallback_supabase_similarity(upload_vec, niche, top_k)
    
    # Return None if all methods fail (let caller decide default)
    if score is None:
        logger.warning(f"All similarity methods failed for {niche}, returning None for caller handling")
        return None
    
    return score


def get_index_cache_stats() -> Dict[str, any]:
    """
    Get statistics about the FAISS index cache.
    
    Returns:
        Dictionary with cache statistics
    """
    return get_cache_stats()


if __name__ == "__main__":
    # Test the reference library
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Reference Library FAISS Integration...")
    
    # Test loading indices
    for niche in ["tech", "gaming", "education"]:
        result = load_faiss_index(niche)
        if result:
            index, ids = result
            print(f"✅ {niche}: {index.ntotal} vectors loaded")
        else:
            print(f"❌ {niche}: not found")
    
    # Test similarity with random vector
    print("\nTesting similarity search...")
    test_vec = np.random.randn(768).astype(np.float32)
    
    for niche in ["tech", "gaming"]:
        score = get_similarity_score(test_vec, niche)
        print(f"  {niche} similarity percentile: {score:.2f}")
        
        similar = faiss_get_similar_thumbnails(test_vec, niche, top_k=3)
        print(f"  Top similar videos: {len(similar)}")
    
    # Cache stats
    stats = get_index_cache_stats()
    print(f"\nCache stats: {stats}")

