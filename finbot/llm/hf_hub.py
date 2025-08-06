"""
HuggingFace Hub Backend Implementation

Provides LLM inference using HuggingFace transformers library
with models loaded from the HuggingFace Hub.
"""

from transformers.pipelines import pipeline
from .base import BaseLLM
from finbot.config import HF_MODEL


class HFHubLLM(BaseLLM):
    """HuggingFace Hub implementation for transformer-based LLM inference."""
    
    def __init__(self):
        """
        Initialize HuggingFace text generation pipeline.
        
        Note: This implementation simulates streaming since HF pipelines
        don't support true streaming by default.
        """
        self.pipeline = pipeline(
            "text-generation",
            model=HF_MODEL,
            device_map="auto",
            max_new_tokens=256
        )

    def stream(self, prompt: str, **kwargs):
        """
        Generate response using HuggingFace pipeline.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for generation
            
        Yields:
            Generated text (simulated streaming)
            
        Note:
            HuggingFace pipelines don't support true streaming, so this
            method generates the full response and yields it as a single token.
        """
        try:
            # Generate full response (HF pipeline limitation)
            result = self.pipeline(prompt, do_sample=False)[0]["generated_text"]
            
            # Extract only the new text (remove input prompt)
            new_text = result[len(prompt):]
            yield new_text
            
        except Exception as e:
            yield f"Error: {str(e)}"
