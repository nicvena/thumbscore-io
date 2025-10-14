"""
FAISS Index Memory Cache

Provides in-memory caching of FAISS indices for instant similarity lookups.
All indices are preloaded on startup and kept in memory for maximum performance.
"""

import os
import logging
import glob
from typing import Dict, Optional, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Global cache for FAISS indices
INDEX_CACHE: Dict[str, faiss.IndexFlatIP] = {}
INDEX_IDS_CACHE: Dict[str, np.ndarray] = {}
INDEX_STATS: Dict[str, int] = {}

def load_indices(path: str = "faiss_indices") -> None:
    """
    Load all FAISS indices from disk into memory cache.
    
    Args:
        path: Directory containing .index files
    """
    global INDEX_CACHE, INDEX_IDS_CACHE, INDEX_STATS
    
    logger.info(f"[FAISS] Loading indices from {path}")
    
    if not os.path.exists(path):
        logger.warning(f"[FAISS] Index directory {path} does not exist")
        return
    
    # Find all .index files
    index_files = glob.glob(os.path.join(path, "*.index"))
    
    if not index_files:
        logger.warning(f"[FAISS] No index files found in {path}")
        return
    
    loaded_count = 0
    total_items = 0
    
    for index_file in index_files:
        # Extract niche name from filename (e.g., "tech.index" -> "tech")
        niche = os.path.basename(index_file).replace(".index", "")
        ids_file = os.path.join(path, f"{niche}_ids.npy")
        
        try:
            # Load FAISS index
            index = faiss.read_index(index_file)
            INDEX_CACHE[niche] = index
            
            # Load corresponding IDs if available
            if os.path.exists(ids_file):
                ids = np.load(ids_file)
                INDEX_IDS_CACHE[niche] = ids
            else:
                logger.warning(f"[FAISS] No IDs file found for {niche}")
                INDEX_IDS_CACHE[niche] = np.array([])
            
            # Store stats
            item_count = index.ntotal
            INDEX_STATS[niche] = item_count
            total_items += item_count
            
            logger.info(f"[FAISS] Loaded index for {niche} ({item_count} items)")
            loaded_count += 1
            
        except Exception as e:
            logger.error(f"[FAISS] Failed to load index for {niche}: {e}")
            continue
    
    logger.info(f"[FAISS] Cache ready with {loaded_count} niches ({total_items} total items)")
    logger.info(f"[FAISS] Memory usage: ~{total_items * 512 * 4 / 1024 / 1024:.1f} MB")

def get_index(niche: str) -> Optional[Tuple[faiss.IndexFlatIP, np.ndarray]]:
    """
    Get cached FAISS index and IDs for a niche.
    
    Args:
        niche: The niche name (e.g., "tech", "gaming")
        
    Returns:
        Tuple of (index, ids) or None if not found
    """
    global INDEX_CACHE, INDEX_IDS_CACHE
    
    if niche not in INDEX_CACHE:
        logger.warning(f"[FAISS] Index for {niche} not found in cache")
        return None
    
    index = INDEX_CACHE[niche]
    ids = INDEX_IDS_CACHE.get(niche, np.array([]))
    
    return index, ids

def get_cache_stats() -> Dict[str, any]:
    """
    Get statistics about the current cache.
    
    Returns:
        Dictionary with cache statistics
    """
    global INDEX_CACHE, INDEX_STATS
    
    stats = {
        "cached_niches": list(INDEX_CACHE.keys()),
        "total_niches": len(INDEX_CACHE),
        "total_items": sum(INDEX_STATS.values()),
        "niche_stats": INDEX_STATS.copy(),
        "memory_usage_mb": sum(INDEX_STATS.values()) * 512 * 4 / 1024 / 1024
    }
    
    return stats

def refresh_indices(path: str = "faiss_indices") -> None:
    """
    Refresh the cache by reloading all indices from disk.
    Used after index rebuild operations.
    
    Args:
        path: Directory containing .index files
    """
    global INDEX_CACHE, INDEX_IDS_CACHE, INDEX_STATS
    
    logger.info("[FAISS] Refreshing index cache...")
    
    # Clear existing cache
    INDEX_CACHE.clear()
    INDEX_IDS_CACHE.clear()
    INDEX_STATS.clear()
    
    # Reload all indices
    load_indices(path)

def clear_cache() -> None:
    """
    Clear all cached indices from memory.
    """
    global INDEX_CACHE, INDEX_IDS_CACHE, INDEX_STATS
    
    logger.info("[FAISS] Clearing index cache...")
    
    INDEX_CACHE.clear()
    INDEX_IDS_CACHE.clear()
    INDEX_STATS.clear()
    
    logger.info("[FAISS] Cache cleared")

def is_cache_ready() -> bool:
    """
    Check if the cache is ready with indices loaded.
    
    Returns:
        True if cache has indices loaded, False otherwise
    """
    return len(INDEX_CACHE) > 0

def get_available_niches() -> list:
    """
    Get list of niches available in the cache.
    
    Returns:
        List of niche names
    """
    return list(INDEX_CACHE.keys())
