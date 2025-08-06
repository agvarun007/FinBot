"""
OpenAI API Backend Implementation

Provides LLM inference using OpenAI's API with proper error handling
and streaming support for real-time responses.
"""

import os
from typing import Optional

from openai import OpenAI
from .base import BaseLLM
from finbot.config import OPENAI_MODEL, STOP_SEQUENCE


class OpenAILLM(BaseLLM):
    """OpenAI API implementation for cloud-based LLM inference."""
    
    def __init__(self):
        """
        Initialize OpenAI client with API key validation.
        
        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

    def stream(self, prompt: str, **kwargs):
        """
        Generate streaming response using OpenAI's chat completion API.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for the API call
            
        Yields:
            Generated text tokens as strings
        """
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                stop=STOP_SEQUENCE,
                stream=True,
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
