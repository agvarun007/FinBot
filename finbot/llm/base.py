"""
Base Language Model Interface

Defines the abstract interface that all LLM backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLM(ABC):
    """Abstract base class for all language model implementations."""
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Generate streaming response tokens for the given prompt.
        
        Args:
            prompt: Input text prompt for the model
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Yields:
            Individual response tokens as strings
        """
        pass
