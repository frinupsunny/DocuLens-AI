"""
End-to-End Integration Test for DocuLens AI.
Tests the full RAG pipeline using real PDF parsing, real Hugging Face embeddings,
FAISS vector indexing, and query retrieval.
"""

import io
import pytest
from pypdf import PdfWriter
from src.pdf_processor import PDFProcessor
from src.text_chunker import TextChunker
from src.embeddings import get_embedding_model
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline


def create_synthetic_pdf() -> io.BytesIO:
    """
    Generate a 2-page valid PDF using raw postscript stream syntax.
    """
    raw_pdf = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 135 >> stream
BT
/F1 12 Tf
50 720 Td
(DocuLens AI Executive Summary: The platform leverages state-of-the-art RAG technology.) Tj
0 -25 Td
(It eliminates hallucinations by enforcing strict context grounding.) Tj
ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
6 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 7 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
7 0 obj << /Length 130 >> stream
BT
/F1 12 Tf
50 720 Td
(Performance Benchmark: FAISS index retrieval latency is under 3 milliseconds.) Tj
0 -25 Td
(The model achieved an exact match accuracy score of 98.4 percent.) Tj
ET
endstream
endobj
xref
0 8
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000122 00000 n 
0000000273 00000 n 
0000000459 00000 n 
0000000536 00000 n 
0000000687 00000 n 
trailer << /Size 8 /Root 1 0 R >>
startxref
868
%%EOF
"""
    return io.BytesIO(raw_pdf)


def test_full_rag_pipeline_end_to_end():
    """
    Verify complete flow:
    PDF Upload -> Text Extraction -> Chunking -> Hugging Face Embeddings -> FAISS Index -> Retrieval
    """
    # 1. Ingest PDF
    pdf_stream = create_synthetic_pdf()
    processor = PDFProcessor()
    docs = processor.extract_documents_from_file(pdf_stream, "benchmark_report.pdf")

    assert len(docs) == 2
    assert docs[0].metadata["page"] == 1
    assert "Executive Summary" in docs[0].page_content
    assert docs[1].metadata["page"] == 2
    assert "Performance Benchmark" in docs[1].page_content

    # 2. Chunk text
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_documents(docs)
    assert len(chunks) >= 2

    # 3. Generate Real Embeddings (Hugging Face all-MiniLM-L6-v2)
    embeddings = get_embedding_model()
    assert embeddings is not None

    # 4. Index in FAISS
    vector_manager = VectorStoreManager(embeddings)
    vector_manager.build_from_documents(chunks)
    assert vector_manager.is_initialized
    assert vector_manager.total_chunks_indexed >= 2

    # 5. Semantic Search: Query for page 2 metrics
    results = vector_manager.similarity_search_with_scores(
        "What is the exact match accuracy score and latency?",
        k=1
    )
    assert len(results) == 1
    top_doc, score = results[0]

    # Verify retrieval accuracy
    assert top_doc.metadata["source"] == "benchmark_report.pdf"
    assert top_doc.metadata["page"] == 2
    assert "98.4 percent" in top_doc.page_content

    # 6. Test RAG Pipeline Execution
    rag = RAGPipeline(vector_store_manager=vector_manager, api_key="")
    response = rag.answer_question("What is the exact match accuracy score?")

    assert response["has_sources"] is True
    assert len(response["sources"]) >= 1
    assert response["sources"][0]["page"] == 2
    assert "98.4 percent" in response["sources"][0]["content"]
