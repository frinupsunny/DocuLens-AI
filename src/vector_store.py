"""
DocuLens AI - FAISS Vector Store Module
Manages vector storage, indexing, persistence, and similarity search
using Facebook AI Similarity Search (FAISS) and local embeddings.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages the FAISS vector database lifecycle:
    - Creation from chunks
    - Incremental addition of newly uploaded documents
    - Top-K similarity search with relevance scoring
    - Persistence and loading from disk
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self.indexed_files: set[str] = set()
        self.total_chunks_indexed: int = 0

    @property
    def is_initialized(self) -> bool:
        """Returns True if the vector store has been populated with chunks."""
        return self.vector_store is not None and self.total_chunks_indexed > 0

    def build_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Build a brand new FAISS vector store from a list of Document chunks.
        """
        if not documents:
            raise ValueError("Cannot build vector store from an empty list of documents.")

        logger.info(f"Indexing {len(documents)} document chunks into FAISS...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        self.indexed_files = {
            doc.metadata.get("source", "unknown") for doc in documents
        }
        self.total_chunks_indexed = len(documents)
        logger.info(
            f"FAISS index created successfully with {self.total_chunks_indexed} chunks "
            f"from {len(self.indexed_files)} files."
        )
        return self.vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """
        Incrementally add newly uploaded document chunks into the active FAISS index.
        """
        if not documents:
            return

        if self.vector_store is None:
            self.build_from_documents(documents)
            return

        logger.info(f"Adding {len(documents)} additional chunks to existing FAISS index...")
        self.vector_store.add_documents(documents)

        for doc in documents:
            source = doc.metadata.get("source")
            if source:
                self.indexed_files.add(source)

        self.total_chunks_indexed += len(documents)
        logger.info(f"FAISS index now contains {self.total_chunks_indexed} total chunks.")

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = config.DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search on the FAISS index and return documents with scores.
        For L2 distance, lower score means closer match.
        For Cosine similarity on normalized vectors, FAISS inner product / cosine distance is used.
        """
        if not self.is_initialized:
            logger.warning("Vector store is not initialized. No documents to search.")
            return []

        if not query or not query.strip():
            return []

        results = self.vector_store.similarity_search_with_score(
            query=query.strip(),
            k=k
        )
        return results

    def similarity_search(
        self,
        query: str,
        k: int = config.DEFAULT_TOP_K
    ) -> List[Document]:
        """
        Perform a similarity search returning the top-K most relevant Document objects.
        """
        if not self.is_initialized:
            return []

        return self.vector_store.similarity_search(query=query.strip(), k=k)

    def save_local(self, folder_path: Path) -> None:
        """
        Save the FAISS index and docstore to disk.
        """
        if not self.is_initialized:
            raise ValueError("Cannot save an uninitialized vector store.")

        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(folder))
        logger.info(f"Saved FAISS index to {folder}")

    def load_local(self, folder_path: Path) -> None:
        """
        Load a saved FAISS index from disk.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"FAISS index path does not exist: {folder}")

        self.vector_store = FAISS.load_local(
            folder_path=str(folder),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True  # Required by langchain for local pickle files
        )
        self.total_chunks_indexed = len(self.vector_store.docstore._dict)
        logger.info(f"Loaded FAISS index from {folder} with {self.total_chunks_indexed} chunks.")

    def clear(self) -> None:
        """Reset the vector database state."""
        self.vector_store = None
        self.indexed_files.clear()
        self.total_chunks_indexed = 0
        logger.info("Vector store cleared.")

    def get_stats(self) -> Dict[str, Any]:
        """Return operational metrics for UI display."""
        return {
            "is_ready": self.is_initialized,
            "total_chunks": self.total_chunks_indexed,
            "indexed_files": list(self.indexed_files),
            "file_count": len(self.indexed_files),
        }
