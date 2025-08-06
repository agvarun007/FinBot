"""
Text Chunking Module

Handles splitting large documents into smaller, overlapping chunks
for efficient processing and retrieval.
"""

from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Input text to be chunked
        chunk_size: Number of tokens (words) per chunk
        overlap: Number of tokens to overlap between consecutive chunks
        
    Returns:
        List of text chunks as strings
        
    Note:
        Uses simple whitespace tokenization. For more sophisticated
        tokenization, consider using specialized libraries like tiktoken.
    """
    if not text or not text.strip():
        return []
    
    # Simple whitespace tokenization
    tokens = text.split()
    
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    start_index = 0
    
    while start_index < len(tokens):
        end_index = start_index + chunk_size
        chunk_tokens = tokens[start_index:end_index]
        
        # Join tokens back into text
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start index, accounting for overlap
        start_index += chunk_size - overlap
        
        # Prevent infinite loop if overlap >= chunk_size
        if start_index <= (start_index - chunk_size + overlap):
            break
    
    return chunks