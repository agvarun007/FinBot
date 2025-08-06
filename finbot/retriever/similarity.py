"""
Similarity Retrieval Module

Handles semantic search using cosine similarity on document embeddings
stored in PostgreSQL with pgvector extension.
"""

from typing import List, Dict, Any
import numpy as np

from finbot.db.client import get_connection
from finbot.config import TOP_K
from pgvector.psycopg2.vector import Vector


def retrieve_similar(query_embedding: np.ndarray, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve the most similar document chunks for a given query embedding.
    
    Args:
        query_embedding: NumPy array representing the query's embedding vector
        top_k: Number of most similar chunks to retrieve
        
    Returns:
        List of dictionaries containing chunk data and similarity scores
        Each dict contains: chunk, metadata, source, score
        
    Note:
        Returns empty list if no similar chunks found or on database error
    """
    if query_embedding is None or len(query_embedding) == 0:
        return []
        
    vector = Vector(query_embedding.tolist())
    connection = get_connection()
    cursor = None
    
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT chunk, metadata, source, embedding <=> %s AS distance
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (vector, vector, top_k))
        
        results = cursor.fetchall()
        
        # Format results for consumption
        similar_chunks = []
        for chunk, metadata, source, distance in results:
            similar_chunks.append({
                "chunk": chunk,
                "metadata": metadata,
                "source": source,
                "score": distance
            })
        
        return similar_chunks
        
    except Exception as e:
        print(f"Error retrieving similar chunks: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        connection.close()