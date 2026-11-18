"""
Unit tests for the VectorStoreManager module.
Uses FakeEmbeddings for fast, deterministic, offline testing of FAISS index creation,
document addition, similarity search, and statistics.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from src.vector_store import VectorStoreManager


@pytest.fixture
def fake_embeddings():
    # 384 dimensions matching all-MiniLM-L6-v2
    return FakeEmbeddings(size=384)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Artificial Intelligence and Machine Learning transform enterprise data.",
            metadata={"source": "ai_overview.pdf", "page": 1, "chunk_id": 0}
        ),
        Document(
            page_content="Retrieval-Augmented Generation bridges proprietary data with Large Language Models.",
            metadata={"source": "rag_architecture.pdf", "page": 2, "chunk_id": 1}
        ),
        Document(
            page_content="FAISS provides lightning-fast similarity search over dense vector representations.",
            metadata={"source": "vector_db.pdf", "page": 5, "chunk_id": 2}
        ),
    ]


def test_vector_store_initial_state(fake_embeddings):
    """Verify vector store initial empty state."""
    manager = VectorStoreManager(fake_embeddings)
    assert not manager.is_initialized
    assert manager.total_chunks_indexed == 0
    assert len(manager.indexed_files) == 0

    stats = manager.get_stats()
    assert not stats["is_ready"]
    assert stats["total_chunks"] == 0


def test_build_from_documents_success(fake_embeddings, sample_documents):
    """Verify FAISS vector store indexing and query retrieval."""
    manager = VectorStoreManager(fake_embeddings)
    manager.build_from_documents(sample_documents)

    assert manager.is_initialized
    assert manager.total_chunks_indexed == 3
    assert len(manager.indexed_files) == 3
    assert "ai_overview.pdf" in manager.indexed_files

    # Search
    results = manager.similarity_search("Artificial Intelligence", k=2)
    assert len(results) == 2
    assert isinstance(results[0], Document)


def test_add_documents_incremental(fake_embeddings, sample_documents):
    """Verify incremental indexing of newly added documents."""
    manager = VectorStoreManager(fake_embeddings)
    manager.build_from_documents(sample_documents[:2])
    assert manager.total_chunks_indexed == 2

    # Incrementally add the third document
    manager.add_documents([sample_documents[2]])
    assert manager.total_chunks_indexed == 3
    assert "vector_db.pdf" in manager.indexed_files


def test_empty_documents_error(fake_embeddings):
    """Verify error handling on empty document list."""
    manager = VectorStoreManager(fake_embeddings)
    with pytest.raises(ValueError, match="empty list"):
        manager.build_from_documents([])


def test_clear_vector_store(fake_embeddings, sample_documents):
    """Verify reset function clears all indices and metadata."""
    manager = VectorStoreManager(fake_embeddings)
    manager.build_from_documents(sample_documents)
    assert manager.is_initialized

    manager.clear()
    assert not manager.is_initialized
    assert manager.total_chunks_indexed == 0
    assert len(manager.indexed_files) == 0
