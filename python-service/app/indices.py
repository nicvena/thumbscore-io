"""
FAISS Index Management for Thumbnail Similarity Search

This module handles building and managing FAISS indices for fast similarity search
across different niches of YouTube thumbnails.
"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
INDEX_CACHE_DIR = os.getenv("INDEX_CACHE_DIR", "./indices")
EMBEDDING_DIM = 768  # CLIP ViT-L/14 embedding dimension

# Niche categories
NICHES = ["tech", "gaming", "education", "entertainment", "people"]


class FAISSIndexManager:
    """Manages FAISS indices for each niche category."""
    
    def __init__(self):
        """Initialize the index manager."""
        if not faiss:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.indices: Dict[str, faiss.Index] = {}
        self.metadata: Dict[str, List[Dict]] = {}
        
        # Create cache directory
        os.makedirs(INDEX_CACHE_DIR, exist_ok=True)
        
        logger.info(f"FAISS Index Manager initialized with cache dir: {INDEX_CACHE_DIR}")
    
    def _get_index_path(self, niche: str) -> str:
        """Get the file path for a niche's index."""
        return os.path.join(INDEX_CACHE_DIR, f"{niche}_index.faiss")
    
    def _get_metadata_path(self, niche: str) -> str:
        """Get the file path for a niche's metadata."""
        return os.path.join(INDEX_CACHE_DIR, f"{niche}_metadata.pkl")
    
    def _fetch_embeddings_from_db(self, niche: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Fetch embeddings and metadata from Supabase for a specific niche.
        
        Args:
            niche: Niche category name
            
        Returns:
            Tuple of (embeddings_array, metadata_list)
        """
        logger.info(f"Fetching embeddings for niche: {niche}")
        
        try:
            # Fetch embeddings from Supabase
            result = self.supabase.table("ref_thumbnails").select(
                "video_id, title, thumbnail_url, views_per_hour, embedding, collected_at"
            ).eq("niche", niche).execute()
            
            if not result.data:
                logger.warning(f"No embeddings found for niche: {niche}")
                return np.array([]).reshape(0, EMBEDDING_DIM), []
            
            embeddings = []
            metadata = []
            
            for row in result.data:
                # Convert embedding list to numpy array
                embedding = np.array(row["embedding"], dtype=np.float32)
                embeddings.append(embedding)
                
                # Store metadata for this embedding
                metadata.append({
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "thumbnail_url": row["thumbnail_url"],
                    "views_per_hour": row["views_per_hour"],
                    "collected_at": row["collected_at"]
                })
            
            embeddings_array = np.vstack(embeddings)
            logger.info(f"Fetched {len(embeddings)} embeddings for {niche}")
            
            return embeddings_array, metadata
            
        except Exception as e:
            logger.error(f"Error fetching embeddings for {niche}: {e}")
            return np.array([]).reshape(0, EMBEDDING_DIM), []
    
    def build_index(self, niche: str) -> bool:
        """
        Build FAISS index for a specific niche.
        
        Args:
            niche: Niche category name
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Building FAISS index for niche: {niche}")
        
        try:
            # Fetch embeddings from database
            embeddings, metadata = self._fetch_embeddings_from_db(niche)
            
            if len(embeddings) == 0:
                logger.warning(f"No embeddings to index for {niche}")
                return False
            
            # Create FAISS index
            # Using IndexFlatIP for cosine similarity (inner product)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            index.add(embeddings)
            
            # Store index and metadata
            self.indices[niche] = index
            self.metadata[niche] = metadata
            
            # Save to disk
            self._save_index(niche)
            
            logger.info(f"Built FAISS index for {niche}: {index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error building index for {niche}: {e}")
            return False
    
    def _save_index(self, niche: str) -> bool:
        """
        Save index and metadata to disk.
        
        Args:
            niche: Niche category name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index_path = self._get_index_path(niche)
            metadata_path = self._get_metadata_path(niche)
            
            # Save FAISS index
            faiss.write_index(self.indices[niche], index_path)
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata[niche], f)
            
            logger.info(f"Saved index and metadata for {niche}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index for {niche}: {e}")
            return False
    
    def _load_index(self, niche: str) -> bool:
        """
        Load index and metadata from disk.
        
        Args:
            niche: Niche category name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index_path = self._get_index_path(niche)
            metadata_path = self._get_metadata_path(niche)
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                logger.warning(f"Index files not found for {niche}")
                return False
            
            # Load FAISS index
            self.indices[niche] = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata[niche] = pickle.load(f)
            
            logger.info(f"Loaded index for {niche}: {self.indices[niche].ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index for {niche}: {e}")
            return False
    
    def rebuild_all_indices(self) -> Dict[str, bool]:
        """
        Rebuild FAISS indices for all niches.
        
        Returns:
            Dictionary mapping niche names to success status
        """
        logger.info("Rebuilding all FAISS indices...")
        start_time = datetime.now()
        
        results = {}
        
        for niche in NICHES:
            logger.info(f"Processing niche: {niche}")
            success = self.build_index(niche)
            results[niche] = success
        
        duration = (datetime.now() - start_time).total_seconds()
        successful = sum(results.values())
        
        logger.info(f"Index rebuilding completed in {duration:.2f}s: {successful}/{len(NICHES)} successful")
        
        return results
    
    def search_similar(self, query_embedding: np.ndarray, niche: str, k: int = 10) -> List[Dict]:
        """
        Search for similar thumbnails in a specific niche.
        
        Args:
            query_embedding: Query embedding vector
            niche: Niche category to search in
            k: Number of similar results to return
            
        Returns:
            List of similar thumbnail metadata
        """
        if niche not in self.indices:
            # Try to load from disk
            if not self._load_index(niche):
                logger.error(f"No index available for niche: {niche}")
                return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.indices[niche].search(query_embedding, k)
            
            # Get metadata for results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata[niche]):
                    result = self.metadata[niche][idx].copy()
                    result["similarity_score"] = float(score)
                    results.append(result)
            
            logger.debug(f"Found {len(results)} similar thumbnails for {niche}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in {niche}: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all indices.
        
        Returns:
            Dictionary with index statistics
        """
        stats = {}
        
        for niche in NICHES:
            try:
                if niche in self.indices:
                    index = self.indices[niche]
                    stats[niche] = {
                        "vectors": index.ntotal,
                        "dimension": index.d,
                        "loaded": True
                    }
                elif self._load_index(niche):
                    index = self.indices[niche]
                    stats[niche] = {
                        "vectors": index.ntotal,
                        "dimension": index.d,
                        "loaded": True
                    }
                else:
                    stats[niche] = {
                        "vectors": 0,
                        "dimension": EMBEDDING_DIM,
                        "loaded": False
                    }
            except Exception as e:
                logger.error(f"Error getting stats for {niche}: {e}")
                stats[niche] = {
                    "vectors": 0,
                    "dimension": EMBEDDING_DIM,
                    "loaded": False,
                    "error": str(e)
                }
        
        return stats


# Global index manager instance
index_manager: Optional[FAISSIndexManager] = None

def get_index_manager() -> FAISSIndexManager:
    """Get or create the global index manager instance."""
    global index_manager
    if index_manager is None:
        index_manager = FAISSIndexManager()
    return index_manager

def rebuild_indices_sync() -> Dict[str, bool]:
    """
    Synchronous wrapper for rebuilding indices (for scheduler compatibility).
    
    Returns:
        Dictionary mapping niche names to success status
    """
    manager = get_index_manager()
    return manager.rebuild_all_indices()
