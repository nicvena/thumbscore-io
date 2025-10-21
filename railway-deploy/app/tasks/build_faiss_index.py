"""
FAISS Index Builder for Thumbnail Lab

This module builds FAISS indices for each niche after the nightly library refresh.
Enables instant similarity lookups in the /v1/score endpoint.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import faiss
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_indices")
EMBEDDING_DIM = 768  # CLIP ViT-L/14 dimension

# Niche categories
NICHES = ["tech", "gaming", "education", "entertainment", "people", "business"]


class FAISSIndexBuilder:
    """Builds and manages FAISS indices for thumbnail similarity search."""
    
    def __init__(self):
        """Initialize the index builder."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        # Create Supabase client (v2.22.0+ doesn't support proxy parameter)
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Create index directory
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        logger.info(f"FAISS index path: {FAISS_INDEX_PATH}")
    
    def _get_index_path(self, niche: str) -> str:
        """Get the file path for a niche's FAISS index."""
        return os.path.join(FAISS_INDEX_PATH, f"{niche}.index")
    
    def _get_metadata_path(self, niche: str) -> str:
        """Get the file path for a niche's metadata (video IDs)."""
        return os.path.join(FAISS_INDEX_PATH, f"{niche}_ids.npy")
    
    def fetch_niche_embeddings(self, niche: str) -> tuple[np.ndarray, List[str]]:
        """
        Fetch embeddings and video IDs from Supabase for a specific niche.
        
        Args:
            niche: Niche category name
            
        Returns:
            Tuple of (embeddings_array, video_ids_list)
        """
        try:
            logger.info(f"Fetching embeddings for niche: {niche}")
            
            # Query Supabase for embeddings
            result = self.supabase.table("ref_thumbnails").select(
                "video_id, embedding"
            ).eq("niche", niche).order("views_per_hour", desc=True).execute()
            
            if not result.data or len(result.data) == 0:
                logger.warning(f"No embeddings found for niche: {niche}")
                return np.array([]).reshape(0, EMBEDDING_DIM), []
            
            embeddings = []
            video_ids = []
            
            for row in result.data:
                embedding = np.array(row["embedding"], dtype=np.float32)
                embeddings.append(embedding)
                video_ids.append(row["video_id"])
            
            embeddings_array = np.vstack(embeddings)
            logger.info(f"Fetched {len(embeddings)} embeddings for {niche}")
            
            return embeddings_array, video_ids
            
        except Exception as e:
            logger.error(f"Error fetching embeddings for {niche}: {e}")
            return np.array([]).reshape(0, EMBEDDING_DIM), []
    
    def build_index_for_niche(self, niche: str) -> bool:
        """
        Build FAISS index for a specific niche.
        
        Args:
            niche: Niche category name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch embeddings and IDs
            embeddings, video_ids = self.fetch_niche_embeddings(niche)
            
            if len(embeddings) == 0:
                logger.warning(f"Skipping FAISS index for {niche} - no embeddings")
                return False
            
            # Create FAISS index using IndexFlatIP (inner product / cosine similarity)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            index.add(embeddings)
            
            # Save index to disk
            index_path = self._get_index_path(niche)
            faiss.write_index(index, index_path)
            
            # Save video IDs mapping
            metadata_path = self._get_metadata_path(niche)
            np.save(metadata_path, np.array(video_ids))
            
            logger.info(f"Built FAISS index for {niche} with {index.ntotal} items")
            return True
            
        except Exception as e:
            logger.error(f"Error building index for {niche}: {e}")
            return False
    
    def build_all_indices(self) -> Dict[str, bool]:
        """
        Build FAISS indices for all niches.
        
        Returns:
            Dictionary mapping niche names to success status
        """
        logger.info("Rebuilding FAISS indices...")
        start_time = datetime.now()
        
        results = {}
        
        for niche in NICHES:
            success = self.build_index_for_niche(niche)
            results[niche] = success
        
        duration = (datetime.now() - start_time).total_seconds()
        successful = sum(results.values())
        
        logger.info(f"Done building FAISS indices in {duration:.2f}s: {successful}/{len(NICHES)} successful")
        
        return results
    
    def get_index_info(self) -> Dict[str, Dict]:
        """
        Get information about existing FAISS indices.
        
        Returns:
            Dictionary with index information
        """
        info = {}
        
        for niche in NICHES:
            index_path = self._get_index_path(niche)
            metadata_path = self._get_metadata_path(niche)
            
            if os.path.exists(index_path):
                try:
                    index = faiss.read_index(index_path)
                    info[niche] = {
                        "exists": True,
                        "num_vectors": index.ntotal,
                        "dimension": index.d,
                        "file_size_mb": os.path.getsize(index_path) / (1024 * 1024),
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(index_path)
                        ).isoformat()
                    }
                except Exception as e:
                    info[niche] = {
                        "exists": True,
                        "error": str(e)
                    }
            else:
                info[niche] = {
                    "exists": False,
                    "num_vectors": 0
                }
        
        return info


# Singleton instance
_builder: Optional[FAISSIndexBuilder] = None

def get_builder() -> FAISSIndexBuilder:
    """Get or create the global builder instance."""
    global _builder
    if _builder is None:
        _builder = FAISSIndexBuilder()
    return _builder

def build_faiss_indices() -> Dict[str, bool]:
    """
    Build FAISS indices for all niches.
    Main function to call after library refresh.
    
    Returns:
        Dictionary mapping niche names to success status
    """
    builder = get_builder()
    return builder.build_all_indices()

def get_faiss_index_info() -> Dict[str, Dict]:
    """
    Get information about existing FAISS indices.
    
    Returns:
        Dictionary with index information
    """
    builder = get_builder()
    return builder.get_index_info()


if __name__ == "__main__":
    # Test the builder
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing FAISS Index Builder...")
    results = build_faiss_indices()
    
    print("\nResults:")
    for niche, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {niche}")
    
    print("\nIndex Info:")
    info = get_faiss_index_info()
    for niche, details in info.items():
        if details.get("exists"):
            print(f"  {niche}: {details.get('num_vectors', 0)} vectors")
        else:
            print(f"  {niche}: not found")

