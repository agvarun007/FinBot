"""
Document Ingestion Module

Orchestrates the complete document ingestion pipeline including
loading, chunking, embedding, and storing documents in the database.
"""

from typing import List, Tuple

from finbot.ingestion.loader import load_sources
from finbot.ingestion.chunker import chunk_text
from finbot.embedding.embedder import embed
from finbot.db.client import upsert_chunks
from finbot.config import CHUNK_SIZE, CHUNK_OVERLAP


def ingest(source_directory: str = "data/raw") -> None:
    """
    Process and ingest documents from the specified directory.
    
    This function:
    1. Loads documents from the source directory
    2. Chunks them into smaller pieces
    3. Generates embeddings for each chunk
    4. Stores everything in the database
    
    Args:
        source_directory: Directory containing documents to process
        
    Raises:
        Exception: If ingestion pipeline fails at any stage
    """
    print(f"Loading documents from {source_directory}...")
    document_sources = load_sources(source_directory)
    
    if not document_sources:
        print("No documents found in the specified directory")
        return
    
    print(f"Processing {len(document_sources)} documents...")
    
    # Prepare data structures for batch processing
    chunk_texts = []
    chunk_metadata = []
    
    # Process each document
    for file_path, full_text in document_sources:
        print(f"Processing: {file_path}")
        
        # Split document into manageable chunks
        text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Create metadata for each chunk
        for chunk in text_chunks:
            chunk_metadata.append({
                "source": file_path,
                "chunk": chunk,
                "metadata": {}  # Additional metadata can be added here
            })
            chunk_texts.append(chunk)
    
    if not chunk_texts:
        print("No text chunks generated from documents")
        return
    
    print(f"Generated {len(chunk_texts)} chunks, creating embeddings...")
    
    # Generate embeddings for all chunks
    embeddings = embed(chunk_texts)
    
    print("Storing chunks and embeddings in database...")
    
    # Store in database
    upsert_chunks(chunk_metadata, embeddings)
    
    print(f"Successfully ingested {len(chunk_texts)} chunks from {len(document_sources)} documents")
