"""
DocuLens AI - PDF Processing Module
Extracts, validates, cleans, and pre-processes text from PDF documents.
Preserves page numbers, file names, and document metadata for RAG citations.
"""

import io
import re
import logging
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document

# Set up module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PDFProcessingError(Exception):
    """Custom exception raised when PDF processing fails."""
    pass


class PDFProcessor:
    """
    Handles PDF validation, text extraction, page-level tracking,
    and text cleaning.
    """

    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    @staticmethod
    def validate_pdf(file_input: Union[str, Path, io.BytesIO], filename: str) -> bool:
        """
        Validate whether the given file is a non-empty, legitimate PDF.
        """
        if not filename.lower().endswith(".pdf"):
            raise PDFProcessingError(f"File '{filename}' is not a PDF.")

        # Check content
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists() or path.stat().st_size == 0:
                raise PDFProcessingError(f"PDF file '{filename}' does not exist or is empty.")
            with open(path, "rb") as f:
                header = f.read(5)
        elif hasattr(file_input, "read"):
            # Seek to start
            file_input.seek(0)
            header = file_input.read(5)
            file_input.seek(0)
            if len(header) == 0:
                raise PDFProcessingError(f"PDF stream for '{filename}' is completely empty.")
        else:
            raise PDFProcessingError("Unsupported file input type.")

        if not header.startswith(b"%PDF-"):
            raise PDFProcessingError(f"File '{filename}' lacks a valid PDF header.")

        return True

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Clean and normalize raw extracted text:
        - Replaces non-breaking spaces and irregular unicode whitespaces
        - Collapses repeated spaces and tabs
        - Normalizes multiple blank lines into maximum 2 newlines
        - Strips extraneous leading/trailing whitespace
        """
        if not raw_text or not isinstance(raw_text, str):
            return ""

        # Replace carriage returns and unicode form feeds
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")

        # Replace non-breaking spaces
        text = text.replace("\u00a0", " ")

        # Remove null and non-printable control characters (except standard newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Collapse horizontal whitespace (spaces, tabs) into single spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        return text.strip()

    def extract_documents_from_file(
        self,
        file_input: Union[str, Path, io.BytesIO],
        filename: str
    ) -> List[Document]:
        """
        Extract page-level LangChain Documents from a PDF file.
        Each Document contains:
          - page_content: Cleaned text from that page
          - metadata: {
                'source': filename,
                'page': page_number (1-indexed),
                'total_pages': total_pages
            }
        """
        self.validate_pdf(file_input, filename)

        try:
            reader = PdfReader(file_input)
        except Exception as e:
            logger.error(f"Error opening PDF '{filename}': {e}")
            raise PDFProcessingError(f"Could not read PDF '{filename}': {str(e)}")

        if reader.is_encrypted:
            try:
                # Attempt decrypt with empty password for unprompted protected files
                reader.decrypt("")
            except Exception:
                raise PDFProcessingError(f"PDF '{filename}' is encrypted and password-protected.")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise PDFProcessingError(f"PDF '{filename}' contains 0 pages.")

        documents: List[Document] = []
        has_any_text = False

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            try:
                raw_page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num} of '{filename}': {e}")
                raw_page_text = ""

            cleaned_page_text = self.clean_text(raw_page_text)
            if cleaned_page_text:
                has_any_text = True

            doc = Document(
                page_content=cleaned_page_text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "total_pages": total_pages,
                }
            )
            documents.append(doc)

        if not has_any_text:
            logger.warning(
                f"PDF '{filename}' contains {total_pages} pages, but no extractable text was found. "
                "It may be a scanned image-only PDF."
            )

        return documents

    def process_multiple_pdfs(
        self,
        pdf_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process multiple PDFs provided as a list of dicts:
        [{ 'file': file_or_path, 'name': filename, 'size': file_size_bytes }]

        Returns a dictionary with:
        - 'documents': All page-level Document objects
        - 'processed_files': List of successfully processed file names
        - 'failed_files': Dict mapping failed file name to error message
        - 'total_pages': Total pages across all processed documents
        """
        all_documents: List[Document] = []
        processed_files: List[str] = []
        failed_files: Dict[str, str] = {}
        total_pages = 0

        for item in pdf_files:
            file_obj = item.get("file")
            filename = item.get("name", "document.pdf")
            try:
                docs = self.extract_documents_from_file(file_obj, filename)
                # Keep non-empty documents for downstream processing
                valid_docs = [d for d in docs if d.page_content.strip()]
                all_documents.extend(valid_docs)
                processed_files.append(filename)
                total_pages += len(docs)
                logger.info(f"Successfully processed '{filename}': {len(docs)} pages.")
            except Exception as e:
                logger.error(f"Failed to process '{filename}': {str(e)}")
                failed_files[filename] = str(e)

        return {
            "documents": all_documents,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "total_pages": total_pages,
        }
