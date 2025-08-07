"""
OpenAI API Backend Implementation

Provides LLM inference using OpenAI's API with proper error handling
and streaming support for real-time responses.
"""

import os
from openai import OpenAI
from .base import BaseLLM
from finbot.config import OPENAI_MODEL, STOP_SEQUENCE, LLM_TEMPERATURE


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
        client = OpenAI(api_key=api_key)

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
            response = client.chat.completions.create(# type: ignore
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                stop=STOP_SEQUENCE,
                stream=True)
            for chunk in response:
                content = chunk.choices[0].delta.get('content', '')
                if content:
                    yield content
        except Exception as e:
            yield f"Error: {str(e)}"
