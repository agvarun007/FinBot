"""
Prompt Formatting Module

Builds optimized prompts for the Language Model using retrieved context
and user queries. Formats prompts according to the Llama-3 chat template
for better instruction following.
"""

from typing import List, Dict, Optional


def build_prompt(
    query: str, 
    retrieved_chunks: List[Dict], 
    system_instructions: Optional[str] = None
) -> str:
    """
    Build a structured prompt for the language model.
    
    Args:
        query: The user's question
        retrieved_chunks: List of relevant document chunks from retrieval
        system_instructions: Optional custom system prompt
    
    Returns:
        Formatted prompt string ready for the language model
    """
    # Define system instructions for the financial assistant
    system = system_instructions.strip() if system_instructions else (
        "You are a helpful Canadian financial expert assistant. "
        "Answer the user's question using the provided context. "
        "Give a clear, concise answer. If the answer isn't in the context, "
        "say 'I don't have that information in the provided documents.'"
    )

    # Process and limit context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks[:3]):  # Limit to top 3 chunks for performance
        chunk_text = chunk.get("chunk", "").strip()
        if chunk_text and len(chunk_text) > 50:  # Filter out very short chunks
            # Truncate excessively long chunks to prevent context overflow
            if len(chunk_text) > 400:
                chunk_text = chunk_text[:400] + "..."
            context_parts.append(chunk_text)

    context = "\n\n".join(context_parts)

    # Format using Llama-3 chat template for optimal performance
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt