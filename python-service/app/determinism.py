"""
Deterministic scoring utilities for Thumbscore.io

Ensures identical thumbnails always return the same scores unless model version changes.
Implements hash-based caching, deterministic seeds, and versioned scoring metadata.
"""

import os
import random
import numpy as np
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# DETERMINISTIC SEEDS
# ============================================================================

def set_deterministic_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for deterministic behavior across the entire pipeline.
    
    Args:
        seed: Random seed value (default: 42)
    """
    logger.info(f"[DETERMINISTIC] Setting seeds to {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("[DETERMINISTIC] PyTorch seeds set")
    except ImportError:
        logger.debug("[DETERMINISTIC] PyTorch not available, skipping")
    except Exception as e:
        logger.warning(f"[DETERMINISTIC] PyTorch seed setting failed: {e}")

# ============================================================================
# HASH-BASED CACHING
# ============================================================================

class DeterministicCache:
    """
    Hash-based cache for embeddings and scores to ensure deterministic results.
    """
    
    def __init__(self, cache_dir: str = "deterministic_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_cache = {}
        self.score_cache = {}
        logger.info(f"[DETERMINISTIC] Cache initialized at {self.cache_dir}")
    
    def _generate_image_hash(self, image_data: bytes) -> str:
        """Generate SHA256 hash of image data."""
        return hashlib.sha256(image_data).hexdigest()
    
    def _generate_request_hash(self, image_hash: str, niche: str, model_version: str) -> str:
        """Generate hash for caching requests."""
        request_data = {
            "image_hash": image_hash,
            "niche": niche,
            "model_version": model_version
        }
        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    def get_cached_embedding(self, image_data: bytes, niche: str, model_version: str) -> Optional[np.ndarray]:
        """
        Get cached CLIP embedding if available.
        
        Args:
            image_data: Raw image bytes
            niche: Content niche (gaming, tech, etc.)
            model_version: Model version string
            
        Returns:
            Cached embedding array or None if not cached
        """
        image_hash = self._generate_image_hash(image_data)
        cache_key = self._generate_request_hash(image_hash, niche, model_version)
        
        cache_file = self.cache_dir / f"embedding_{cache_key}.npy"
        
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                logger.debug(f"[DETERMINISTIC] Cache hit for embedding: {cache_key[:8]}...")
                return embedding
            except Exception as e:
                logger.warning(f"[DETERMINISTIC] Failed to load cached embedding: {e}")
        
        return None
    
    def cache_embedding(self, image_data: bytes, niche: str, model_version: str, embedding: np.ndarray) -> None:
        """
        Cache CLIP embedding for future use.
        
        Args:
            image_data: Raw image bytes
            niche: Content niche
            model_version: Model version string
            embedding: CLIP embedding array
        """
        image_hash = self._generate_image_hash(image_data)
        cache_key = self._generate_request_hash(image_hash, niche, model_version)
        
        cache_file = self.cache_dir / f"embedding_{cache_key}.npy"
        
        try:
            np.save(cache_file, embedding)
            logger.debug(f"[DETERMINISTIC] Cached embedding: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"[DETERMINISTIC] Failed to cache embedding: {e}")
    
    def get_cached_score(self, image_data: bytes, niche: str, model_version: str) -> Optional[Dict[str, Any]]:
        """
        Get cached final score if available.
        
        Args:
            image_data: Raw image bytes
            niche: Content niche
            model_version: Model version string
            
        Returns:
            Cached score dictionary or None if not cached
        """
        image_hash = self._generate_image_hash(image_data)
        cache_key = self._generate_request_hash(image_hash, niche, model_version)
        
        cache_file = self.cache_dir / f"score_{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    score_data = json.load(f)
                logger.debug(f"[DETERMINISTIC] Cache hit for score: {cache_key[:8]}...")
                return score_data
            except Exception as e:
                logger.warning(f"[DETERMINISTIC] Failed to load cached score: {e}")
        
        return None
    
    def cache_score(self, image_data: bytes, niche: str, model_version: str, score_data: Dict[str, Any]) -> None:
        """
        Cache final score for future use.
        
        Args:
            image_data: Raw image bytes
            niche: Content niche
            model_version: Model version string
            score_data: Complete score dictionary
        """
        image_hash = self._generate_image_hash(image_data)
        cache_key = self._generate_request_hash(image_hash, niche, model_version)
        
        cache_file = self.cache_dir / f"score_{cache_key}.json"
        
        try:
            # Create a JSON-serializable copy of score_data
            serializable_data = self._make_json_serializable(score_data)
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            logger.debug(f"[DETERMINISTIC] Cached score: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"[DETERMINISTIC] Failed to cache score: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy arrays and other non-serializable objects to JSON-serializable format.
        
        Args:
            obj: Object to make JSON-serializable
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bytes):
            # Convert bytes to base64 string for JSON serialization
            import base64
            return base64.b64encode(obj).decode('utf-8')
        elif hasattr(obj, 'tolist'):  # Other numpy types
            return obj.tolist()
        else:
            return obj
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        logger.info("[DETERMINISTIC] Cache cleared")

# ============================================================================
# DETERMINISTIC FAISS OPERATIONS
# ============================================================================

def deterministic_faiss_search(index, query_vector: np.ndarray, k: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform deterministic FAISS search with consistent neighbor ordering.
    
    Args:
        index: FAISS index
        query_vector: Query embedding vector
        k: Number of neighbors to return
        
    Returns:
        Tuple of (distances, indices) with consistent ordering
    """
    # Ensure query vector is normalized and deterministic
    query_vector = query_vector.astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Perform search
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    
    # Sort results by distance for deterministic ordering
    # When distances are equal, sort by index for consistency
    sort_indices = np.lexsort((indices[0], distances[0]))
    
    return distances[0][sort_indices], indices[0][sort_indices]

# ============================================================================
# GLOBAL NORMALIZATION
# ============================================================================

class GlobalNormalizer:
    """
    Global normalization for scores across all niches to ensure consistent scaling.
    """
    
    def __init__(self):
        self.niche_stats = {}
        self.global_stats = {}
        logger.info("[DETERMINISTIC] Global normalizer initialized")
    
    def update_niche_stats(self, niche: str, scores: np.ndarray) -> None:
        """
        Update statistics for a specific niche.
        
        Args:
            niche: Content niche
            scores: Array of scores for this niche
        """
        if niche not in self.niche_stats:
            self.niche_stats[niche] = {
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 100.0,
                'count': 0
            }
        
        if len(scores) > 0:
            self.niche_stats[niche]['mean'] = np.mean(scores)
            self.niche_stats[niche]['std'] = np.std(scores)
            self.niche_stats[niche]['min'] = np.min(scores)
            self.niche_stats[niche]['max'] = np.max(scores)
            self.niche_stats[niche]['count'] = len(scores)
    
    def normalize_score(self, score: float, niche: str) -> float:
        """
        Normalize score using global statistics for consistent scaling.
        
        Args:
            score: Raw score to normalize
            niche: Content niche
            
        Returns:
            Normalized score
        """
        if niche not in self.niche_stats:
            # Use default normalization if niche stats not available
            return max(0, min(100, score))
        
        stats = self.niche_stats[niche]
        
        # Z-score normalization with clipping
        if stats['std'] > 0:
            z_score = (score - stats['mean']) / stats['std']
            # Map to 0-100 range
            normalized = 50 + (z_score * 15)  # 15 is scaling factor
            return max(0, min(100, normalized))
        else:
            return max(0, min(100, score))

# ============================================================================
# VERSIONED SCORING METADATA
# ============================================================================

def get_scoring_metadata() -> Dict[str, Any]:
    """
    Get versioned scoring metadata for API responses.
    
    Returns:
        Dictionary with scoring version and metadata
    """
    score_version = os.getenv("SCORE_VERSION", "v1.4-faiss-hybrid")
    deterministic_mode = os.getenv("DETERMINISTIC_MODE", "false").lower() == "true"
    
    metadata = {
        "score_version": score_version,
        "deterministic_mode": deterministic_mode,
        "timestamp": None,  # Will be set by caller
        "model_info": {
            "clip_version": "ViT-L/14",
            "faiss_enabled": True,
            "power_words_version": "v2.0-expanded",
            "amplification_enabled": True
        },
        "cache_info": {
            "embedding_cache_enabled": deterministic_mode,
            "score_cache_enabled": deterministic_mode,
            "cache_version": "v1.0"
        }
    }
    
    return metadata

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_deterministic_mode() -> Tuple[bool, DeterministicCache, GlobalNormalizer]:
    """
    Initialize deterministic mode and return components.
    
    Returns:
        Tuple of (is_deterministic, cache, normalizer)
    """
    deterministic_mode = os.getenv("DETERMINISTIC_MODE", "false").lower() == "true"
    
    if deterministic_mode:
        logger.info("[DETERMINISTIC] Initializing deterministic mode")
        set_deterministic_seeds()
        cache = DeterministicCache()
        normalizer = GlobalNormalizer()
        return True, cache, normalizer
    else:
        logger.info("[DETERMINISTIC] Running in non-deterministic mode")
        return False, None, None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def round_embedding(embedding: np.ndarray, decimals: int = 4) -> np.ndarray:
    """
    Round embedding to specified decimal places for deterministic behavior.
    
    Args:
        embedding: Input embedding array
        decimals: Number of decimal places
        
    Returns:
        Rounded embedding array
    """
    return np.round(embedding, decimals=decimals)

def ensure_deterministic_array(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is deterministic by rounding and setting dtype.
    
    Args:
        arr: Input array
        
    Returns:
        Deterministic array
    """
    arr = arr.astype(np.float32)
    arr = np.round(arr, decimals=4)
    return arr
