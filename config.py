"""
DocuLens AI - Configuration Module
Centralizes all application configurations, model selections, chunking parameters,
and directory paths. Loads environment variables securely using python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent

# Automatically load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# Security & API Key Validation
# ==============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


def validate_api_key(api_key: str = None) -> bool:
    """
    Validate that an OpenAI API key is formatted correctly.
    Supports either the environment variable or a user-provided session key.
    """
    key = api_key or OPENAI_API_KEY
    if not key:
        return False
    # Standard OpenAI keys begin with 'sk-' and are at least 20 chars long
    return key.startswith("sk-") and len(key) >= 20


# ==============================================================================
# Embedding & Vector Database Settings
# ==============================================================================
# all-MiniLM-L6-v2 provides excellent retrieval quality, fast inference on CPU,
# and generates 384-dimensional dense vectors without consuming paid API credits.
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# Text Chunking Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval Settings
DEFAULT_TOP_K = 4

# ==============================================================================
# LLM Settings
# ==============================================================================
# gpt-4o-mini is cost-effective, fast, and highly accurate for RAG synthesis.
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = 0.0  # 0.0 ensures deterministic, hallucination-resistant answers

# ==============================================================================
# Application Directory Paths
# ==============================================================================
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure runtime directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
