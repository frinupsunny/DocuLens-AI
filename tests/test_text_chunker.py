"""
Unit tests for the TextChunker module.
Tests document splitting, chunk size, chunk overlap, metadata preservation, and statistics.
"""

import pytest
from langchain_core.documents import Document
from src.text_chunker import TextChunker


def test_chunker_initialization_validation():
    """Verify that chunk_overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap.*strictly smaller"):
        TextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap.*strictly smaller"):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_split_documents_preserves_metadata():
    """Verify that source and page metadata are retained on every chunk."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)

    long_text = (
        "Retrieval-Augmented Generation (RAG) improves the quality of LLM responses "
        "by grounding the model on external knowledge bases and private documents. "
        "DocuLens AI provides accurate question answering with transparent citations."
    )

    doc = Document(
        page_content=long_text,
        metadata={"source": "research_paper.pdf", "page": 3}
    )

    chunks = chunker.split_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "research_paper.pdf"
        assert chunk.metadata["page"] == 3
        assert "chunk_id" in chunk.metadata
        assert "chunk_char_count" in chunk.metadata
        assert len(chunk.page_content) > 0


def test_split_empty_documents():
    """Verify that passing an empty list returns an empty list without error."""
    chunker = TextChunker()
    chunks = chunker.split_documents([])
    assert chunks == []


def test_get_chunk_statistics():
    """Verify chunk statistics calculations."""
    docs = [
        Document(page_content="A" * 100, metadata={"source": "doc1.pdf", "page": 1}),
        Document(page_content="B" * 200, metadata={"source": "doc1.pdf", "page": 2}),
    ]

    stats = TextChunker.get_chunk_statistics(docs)
    assert stats["total_chunks"] == 2
    assert stats["avg_chunk_length"] == 150.0
    assert stats["min_chunk_length"] == 100
    assert stats["max_chunk_length"] == 200


def test_chunk_overlap_continuity():
    """Verify that successive chunks share overlapping text."""
    chunker = TextChunker(chunk_size=60, chunk_overlap=20)
    text = (
        "The quick brown fox jumps over the lazy dog repeatedly until "
        "the sun sets across the evening sky."
    )
    doc = Document(page_content=text, metadata={"source": "test.pdf", "page": 1})
    chunks = chunker.split_documents([doc])

    assert len(chunks) >= 2
    # Verify first chunk content is under limit
    assert len(chunks[0].page_content) <= 60
