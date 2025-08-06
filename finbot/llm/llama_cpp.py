"""
Llama.cpp Backend Implementation

Provides local LLM inference using the llama.cpp library with GGUF models.
Optimized for performance with reduced context windows and efficient batching.
"""

import os
from typing import Optional

from llama_cpp import Llama
from finbot.config import LLAMA_PATH, STOP_SEQUENCE, MAX_RESPONSE_TOKENS, LLM_TEMPERATURE
from .base import BaseLLM


class LlamaCppLLM(BaseLLM):
    """Local LLM implementation using llama.cpp for GGUF model inference."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the llama.cpp model.
        
        Args:
            model_path: Path to GGUF model file. Uses LLAMA_PATH from config if None.
            
        Raises:
            ValueError: If model path doesn't exist or is not provided.
        """
        model_path = model_path or LLAMA_PATH
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        
        # Initialize with performance-optimized settings
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Reduced context window for faster inference
            n_batch=512,  # Efficient batch processing
            n_threads=4,  # CPU thread limit
            verbose=False,  # Suppress debug output
            use_mmap=True,  # Memory mapping for efficiency
            use_mlock=False,  # Avoid memory locking
        )

    def stream(self, prompt: str, max_tokens: Optional[int] = None, 
               temperature: Optional[float] = None, **kwargs):
        """
        Generate streaming response for the given prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature for randomness control
            **kwargs: Additional parameters passed to llama.cpp
            
        Yields:
            Generated text tokens as strings
        """
        max_tokens = max_tokens or MAX_RESPONSE_TOKENS
        temperature = temperature or LLM_TEMPERATURE
        
        try:
            for token_data in self.llm(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                stop=STOP_SEQUENCE, 
                stream=True,
                top_k=40,  # Vocabulary selection limit
                top_p=0.9,  # Nucleus sampling
                repeat_penalty=1.1,  # Reduce repetition
                echo=False  # Don't echo input prompt
            ):
                # Extract text content from token response
                if isinstance(token_data, dict) and 'choices' in token_data:
                    text = token_data['choices'][0].get('text', '')
                    if text:
                        yield text
                else:
                    yield str(token_data)
        except Exception as e:
            yield f"Error: {str(e)}"
