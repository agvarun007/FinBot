"""
FinBot Configuration Module

Manages environment variables and application settings.
Validates configuration on startup to ensure proper setup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory for relative path calculations
BASE_DIR = Path(__file__).resolve().parent.parent

# Database configuration
DB_URI = os.getenv("DATABASE_URL", "postgresql:///finbot")

# Embedding model configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM backend configuration
# Supported backends: llama_cpp, openai, hf_hub
LLM_BACKEND = os.getenv("LLM_BACKEND", "llama_cpp")

# Backend-specific model paths and settings
LLAMA_PATH = os.getenv("LLAMA_PATH")  # Path to GGUF model file
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HF_MODEL = os.getenv("HF_MODEL")  # HuggingFace model repository name

# Document processing parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Retrieval and response parameters
TOP_K = int(os.getenv("TOP_K", 4))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", 128))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))

# Stop sequences for response generation
STOP_SEQUENCE = ["### Question", "### Answer", "\n\n---", "Source:", "<|eot_id|>", "</s>"]


def validate_config():
    """
    Validate configuration settings and raise errors for invalid setups.
    
    Raises:
        ValueError: If required configuration is missing or invalid.
    """
    errors = []
    
    if LLM_BACKEND == "llama_cpp":
        if not LLAMA_PATH:
            errors.append("LLAMA_PATH is required when using llama_cpp backend")
        elif not Path(LLAMA_PATH).exists():
            errors.append(f"LLAMA_PATH file does not exist: {LLAMA_PATH}")
    
    elif LLM_BACKEND == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY is required when using openai backend")
    
    elif LLM_BACKEND == "hf_hub":
        if not HF_MODEL:
            errors.append("HF_MODEL is required when using hf_hub backend")
    
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"- {err}" for err in errors)
        raise ValueError(error_message)


# Validate configuration on module import
# Can be disabled by setting SKIP_CONFIG_VALIDATION=1
if not os.getenv("SKIP_CONFIG_VALIDATION"):
    validate_config()
