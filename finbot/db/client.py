"""
Database Client Module

Handles PostgreSQL operations with pgvector for storing and retrieving
document embeddings. Provides connection management and data persistence.
"""

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json
from typing import List, Dict, Any
import numpy as np

from finbot.config import DB_URI


def get_connection():
    """
    Create a PostgreSQL connection with pgvector support.
    
    Returns:
        psycopg2 connection object
        
    Raises:
        ConnectionError: If database connection fails
    """
    try:
        connection = psycopg2.connect(DB_URI)
        register_vector(connection)
        return connection
    except psycopg2.Error as e:
        raise ConnectionError(f"Failed to connect to database: {e}")


def upsert_chunks(chunks_with_metadata: List[Dict[str, Any]], embeddings: np.ndarray):
    """
    Insert document chunks and their embeddings into the database.
    
    Args:
        chunks_with_metadata: List of dictionaries containing chunk data
                              Each dict should have keys: source, chunk, metadata
        embeddings: NumPy array of embeddings corresponding to chunks
        
    Raises:
        ValueError: If no chunks provided
        psycopg2.Error: If database operation fails
    """
    if not chunks_with_metadata or len(chunks_with_metadata) == 0:
        raise ValueError("No chunks provided for database insertion")
        
    connection = get_connection()
    cursor = None
    
    try:
        cursor = connection.cursor()
        
        for metadata, embedding in zip(chunks_with_metadata, embeddings):
            cursor.execute("""
                INSERT INTO documents (source, chunk, metadata, embedding)
                VALUES (%s, %s, %s, %s)
            """, (
                metadata["source"], 
                metadata["chunk"], 
                Json(metadata.get("metadata", {})), 
                embedding.tolist()
            ))
        
        connection.commit()
        
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        if cursor:
            cursor.close()
        connection.close()