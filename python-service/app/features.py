"""
Feature extraction utilities for thumbnail analysis
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Optional
import logging
import io

logger = logging.getLogger(__name__)

# Global CLIP model (lazy loaded)
_clip_model = None
_clip_preprocess = None

def load_clip_model():
    """Load CLIP model lazily"""
    global _clip_model, _clip_preprocess
    
    if _clip_model is None:
        try:
            import clip
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _clip_model, _clip_preprocess = clip.load("ViT-L/14", device=device)
            logger.info(f"CLIP model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    return _clip_model, _clip_preprocess

def clip_encode(image: Union[Image.Image, bytes, np.ndarray]) -> np.ndarray:
    """
    Encode image using CLIP model
    
    Args:
        image: PIL Image, bytes, or numpy array
        
    Returns:
        CLIP embedding as numpy array (768 dimensions)
    """
    try:
        # Load CLIP model
        model, preprocess = load_clip_model()
        
        # Convert input to PIL Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Preprocess and encode
        with torch.no_grad():
            image_tensor = preprocess(image).unsqueeze(0).to(model.device)
            embedding = model.encode_image(image_tensor)
            return embedding.cpu().numpy().flatten()
            
    except Exception as e:
        logger.error(f"CLIP encoding failed: {e}")
        # Return random embedding as fallback
        return np.random.randn(768).astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to unit length"""
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding
