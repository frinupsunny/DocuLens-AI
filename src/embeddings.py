"""
DocuLens AI - Embedding Generation Module
Generates dense vector embeddings using Hugging Face Sentence Transformers locally.
Uses the lightweight, fast, and high-quality 'all-MiniLM-L6-v2' model.
Does not require external API calls or incur per-token charges.
"""

import os
import logging
from typing import Optional

# Suppress Windows symlink warning from huggingface_hub on local machines
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

import config

logger = logging.getLogger(__name__)

# Global cache to prevent re-instantiating the model in memory
_CACHED_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_CACHED_MODEL_NAME: Optional[str] = None


def get_embedding_model(
    model_name: str = config.DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu"
) -> HuggingFaceEmbeddings:
    """
    Get or initialize the HuggingFaceEmbeddings model singleton.
    Cached in memory to optimize latency and RAM usage.

    Args:
        model_name: HuggingFace model repo id (default: all-MiniLM-L6-v2)
        device: 'cpu' or 'cuda' (defaults to 'cpu' for stability)

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    global _CACHED_EMBEDDINGS, _CACHED_MODEL_NAME

    if _CACHED_EMBEDDINGS is not None and _CACHED_MODEL_NAME == model_name:
        return _CACHED_EMBEDDINGS

    logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")

    model_kwargs = {"device": device}
    encode_kwargs = {
        "normalize_embeddings": True  # Normalizes vectors for accurate Cosine Similarity
    }

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        _CACHED_EMBEDDINGS = embeddings
        _CACHED_MODEL_NAME = model_name
        logger.info(f"Embedding model '{model_name}' loaded successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        raise RuntimeError(
            f"Could not load Hugging Face embedding model '{model_name}'. "
            f"Details: {str(e)}"
        )
