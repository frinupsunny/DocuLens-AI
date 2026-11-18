"""
Unit tests for the RAGPipeline module.
Tests question validation, uninitialized state handling, missing API key fallbacks,
and context formatting.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings(size=384)


@pytest.fixture
def initialized_vector_manager(fake_embeddings):
    manager = VectorStoreManager(fake_embeddings)
    docs = [
        Document(
            page_content="DocuLens AI uses RAG to answer user questions from PDF documents.",
            metadata={"source": "doculens_whitepaper.pdf", "page": 1, "chunk_id": 0}
        )
    ]
    manager.build_from_documents(docs)
    return manager


def test_empty_question_handling(fake_embeddings):
    """Verify response when question is empty."""
    manager = VectorStoreManager(fake_embeddings)
    pipeline = RAGPipeline(vector_store_manager=manager, api_key="sk-testkey12345678901234567890")

    result = pipeline.answer_question("")
    assert "valid, non-empty question" in result["answer"]
    assert not result["has_sources"]


def test_uninitialized_store_handling(fake_embeddings):
    """Verify response when no documents are indexed."""
    manager = VectorStoreManager(fake_embeddings)
    pipeline = RAGPipeline(vector_store_manager=manager, api_key="sk-testkey12345678901234567890")

    result = pipeline.answer_question("What is DocuLens AI?")
    assert "No documents have been processed yet" in result["answer"]
    assert not result["has_sources"]


def test_missing_api_key_graceful_handling(initialized_vector_manager):
    """Verify graceful fallback with retrieved sources when OpenAI key is missing."""
    pipeline = RAGPipeline(
        vector_store_manager=initialized_vector_manager,
        api_key=""
    )

    result = pipeline.answer_question("How does DocuLens work?")
    assert result["has_sources"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "doculens_whitepaper.pdf"
    assert "OpenAI API Key Required" in result["answer"]


def test_format_context_structure():
    """Verify context formatting includes chunk numbering and metadata."""
    docs = [
        Document(
            page_content="First chunk text.",
            metadata={"source": "report.pdf", "page": 4}
        ),
        Document(
            page_content="Second chunk text.",
            metadata={"source": "notes.pdf", "page": 2}
        )
    ]
    formatted = RAGPipeline._format_context(docs)

    assert "[Chunk 1 | Source: report.pdf | Page: 4]" in formatted
    assert "First chunk text." in formatted
    assert "[Chunk 2 | Source: notes.pdf | Page: 2]" in formatted
    assert "Second chunk text." in formatted
