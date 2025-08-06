"""
Text Embedding Module

Handles text embedding generation using sentence-transformers models.
Provides caching for efficient model reuse across multiple embedding calls.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from finbot.config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Get the sentence transformer model with caching.
    
    Returns:
        Loaded SentenceTransformer model
        
    Note:
        Model is cached after first load for efficiency.
    """
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of text strings.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        NumPy array of embeddings with shape (len(texts), embedding_dim)
        
    Note:
        Embeddings are normalized for cosine similarity calculations.
    """
    if not texts:
        return np.array([])
    
    model = get_embedding_model()
    
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embeddings
