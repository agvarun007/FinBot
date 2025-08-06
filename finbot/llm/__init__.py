"""
Large Language Model (LLM) Backend Module

Provides a unified interface for different LLM backends including
local models (llama.cpp), OpenAI API, and HuggingFace Hub models.
"""

from finbot.config import LLM_BACKEND


def get_llm():
    """
    Factory function to create appropriate LLM instance based on configuration.
    
    Returns:
        LLM instance configured according to LLM_BACKEND setting
        
    Raises:
        ImportError: If required dependencies for the backend are not installed
        ValueError: If backend configuration is invalid
    """
    if LLM_BACKEND == "openai":
        from .openai import OpenAILLM
        return OpenAILLM()
    
    if LLM_BACKEND == "hf_hub":
        from .hf_hub import HFHubLLM
        return HFHubLLM()
    
    # Default to llama_cpp for local model inference
    from .llama_cpp import LlamaCppLLM
    return LlamaCppLLM()
