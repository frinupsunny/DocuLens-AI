"""
Unit tests for the PDFProcessor module.
Tests PDF validation, text extraction, page metadata tracking, and text cleaning.
"""

import io
import pytest
from src.pdf_processor import PDFProcessor, PDFProcessingError

# Minimal valid PDF binary with extractable text for testing
SAMPLE_VALID_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 55 >> stream
BT
/F1 14 Tf
72 712 Td
(DocuLens AI Test Sample Text) Tj
ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000373 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
457
%%EOF
"""


@pytest.fixture
def processor():
    return PDFProcessor()


def test_clean_text_normalizes_whitespace(processor):
    """Test that multiple spaces and tabs are collapsed."""
    raw = "DocuLens    AI   \t  Retrieval    Augmented  Generation"
    cleaned = processor.clean_text(raw)
    assert cleaned == "DocuLens AI Retrieval Augmented Generation"


def test_clean_text_normalizes_newlines(processor):
    """Test that multiple consecutive newlines are normalized to at most two."""
    raw = "Line 1\n\n\n\n\nLine 2\r\n\r\nLine 3"
    cleaned = processor.clean_text(raw)
    assert cleaned == "Line 1\n\nLine 2\n\nLine 3"


def test_clean_text_removes_control_characters(processor):
    """Test that non-printable characters are stripped."""
    raw = "Hello\x00\x08World\x1f!"
    cleaned = processor.clean_text(raw)
    assert cleaned == "HelloWorld!"


def test_validate_pdf_rejects_non_pdf(processor):
    """Test that non-PDF files raise an error."""
    fake_stream = io.BytesIO(b"Not a PDF")
    with pytest.raises(PDFProcessingError, match="not a PDF"):
        processor.validate_pdf(fake_stream, "document.txt")


def test_validate_pdf_rejects_empty_stream(processor):
    """Test that empty PDF files raise an error."""
    empty_stream = io.BytesIO(b"")
    with pytest.raises(PDFProcessingError, match="empty"):
        processor.validate_pdf(empty_stream, "empty.pdf")


def test_validate_pdf_rejects_invalid_header(processor):
    """Test that files without %PDF- header are rejected."""
    corrupt_stream = io.BytesIO(b"INVALID_HEADER_12345")
    with pytest.raises(PDFProcessingError, match="valid PDF header"):
        processor.validate_pdf(corrupt_stream, "corrupt.pdf")


def test_extract_documents_from_valid_pdf(processor):
    """Test extracting text and verifying metadata from a valid PDF."""
    pdf_stream = io.BytesIO(SAMPLE_VALID_PDF)
    docs = processor.extract_documents_from_file(pdf_stream, "sample.pdf")

    assert len(docs) == 1
    assert "DocuLens AI Test Sample Text" in docs[0].page_content
    assert docs[0].metadata["source"] == "sample.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["total_pages"] == 1


def test_process_multiple_pdfs(processor):
    """Test processing multiple files with mixed valid/invalid inputs."""
    pdf1 = {"file": io.BytesIO(SAMPLE_VALID_PDF), "name": "doc1.pdf"}
    pdf2 = {"file": io.BytesIO(b"bad content"), "name": "corrupted.pdf"}

    result = processor.process_multiple_pdfs([pdf1, pdf2])

    assert "doc1.pdf" in result["processed_files"]
    assert "corrupted.pdf" in result["failed_files"]
    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["source"] == "doc1.pdf"
