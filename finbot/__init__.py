"""
FinBot - Canadian Financial Assistant

A privacy-focused, locally-hosted financial assistant that provides accurate
answers about Canadian financial topics using retrieval-augmented generation.

Author: Varun Agarwal
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Varun Agarwal"
__email__ = "agvarun34@gmail.com"
__description__ = "Privacy-focused Canadian financial assistant using RAG"

# Core module imports for easy access
from finbot.config import validate_config
from finbot.llm import get_llm
from finbot.embedding.embedder import embed
from finbot.retriever.similarity import retrieve_similar
from finbot.prompt.formatter import build_prompt

__all__ = [
    "validate_config",
    "get_llm", 
    "embed",
    "retrieve_similar",
    "build_prompt"
]