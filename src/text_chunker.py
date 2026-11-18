"""
DocuLens AI - Text Chunker Module
Splits extracted document text into semantically coherent, overlapping chunks.
Maintains exact metadata (source document name, page number, chunk id) for citations.
"""

import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Recursively uses natural text boundaries: paragraphs -> sentences -> words -> characters.
    """

    def __init__(
        self,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        separators: List[str] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Standard hierarchical separators for rich natural text
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of page-level Documents into smaller chunks.
        Augments metadata with chunk_id and preserves source and page.
        """
        if not documents:
            return []

        chunks: List[Document] = []
        raw_chunks = self.splitter.split_documents(documents)

        # Enhance chunk metadata with a unique chunk identifier
        for idx, chunk in enumerate(raw_chunks):
            # Clean any leading or trailing whitespace inside chunk
            clean_content = chunk.page_content.strip()
            if not clean_content:
                continue

            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = idx
            metadata["chunk_char_count"] = len(clean_content)

            chunk_doc = Document(
                page_content=clean_content,
                metadata=metadata
            )
            chunks.append(chunk_doc)

        logger.info(
            f"Split {len(documents)} document pages into {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return chunks

    @staticmethod
    def get_chunk_statistics(chunks: List[Document]) -> Dict[str, Any]:
        """
        Compute descriptive statistics for the generated text chunks.
        Helpful for monitoring document processing quality in the UI.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
            }

        lengths = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": round(sum(lengths) / len(lengths), 1),
            "min_chunk_length": min(lengths),
            "max_chunk_length": max(lengths),
        }
