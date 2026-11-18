"""
DocuLens AI - RAG Pipeline Module
Orchestrates context retrieval, prompt construction, LLM invocation via OpenAI,
and citation synthesis with hallucination guardrails.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager
import config

logger = logging.getLogger(__name__)

# Strict System Prompt to eliminate hallucinations and enforce context grounding
RAG_SYSTEM_PROMPT = """You are DocuLens AI, an expert, precise, and professional document analysis assistant.

CRITICAL INSTRUCTIONS:
1. Answer the user's question using ONLY the provided Document Context below.
2. If the answer is not contained within the provided context, state clearly and politely:
   "I could not find information regarding this in the uploaded document(s). Please verify your document or ask a question directly related to its contents."
3. DO NOT extrapolate, assume, or fabricate any information. Never guess or hallucinate.
4. Keep your answer concise, structured, professional, and directly focused on the user's question.
5. If the context contains relevant page or section references, cite them in your response.

---
DOCUMENT CONTEXT:
{context}
---
"""

USER_PROMPT_TEMPLATE = """User Question: {question}

Provide an accurate, context-grounded answer based strictly on the document context above."""


class RAGPipeline:
    """
    Executes the end-to-end Retrieval-Augmented Generation workflow:
    1. Embeds question & retrieves top-K chunks from FAISS
    2. Builds formatted context with metadata tags
    3. Prompts OpenAI Chat model
    4. Formats answer with exact page citations and source snippets
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        api_key: Optional[str] = None,
        model_name: str = config.DEFAULT_LLM_MODEL,
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_k: int = config.DEFAULT_TOP_K
    ):
        self.vector_store_manager = vector_store_manager
        self.api_key = (api_key or config.OPENAI_API_KEY).strip()
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k

        self._llm: Optional[ChatOpenAI] = None
        self._prompt_template = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE)
        ])

    def _get_llm(self) -> ChatOpenAI:
        """Instantiate or retrieve the ChatOpenAI client."""
        if not self.api_key or not config.validate_api_key(self.api_key):
            raise ValueError(
                "OpenAI API key is missing or invalid. Please provide a valid key in the sidebar or in your .env file."
            )

        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key,
                max_retries=2,
            )
        return self._llm

    def update_api_key(self, api_key: str) -> None:
        """Update API key dynamically if user supplies it via the UI."""
        self.api_key = api_key.strip()
        self._llm = None  # Reset client to reinitialize with new key

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        """
        Format retrieved document chunks into clean delimited text
        annotated with filename and page metadata.
        """
        formatted_parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "Unknown Page")
            snippet = doc.page_content.strip()

            formatted_parts.append(
                f"[Chunk {i} | Source: {source} | Page: {page}]\n{snippet}"
            )
        return "\n\n".join(formatted_parts)

    def answer_question(
        self,
        question: str,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute RAG for a given user question.

        Returns:
            Dict containing:
            - 'answer': Generated response text
            - 'sources': List of source dictionaries (filename, page, excerpt, score)
            - 'has_sources': True if relevant chunks were retrieved
            - 'error': Optional error message string
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid, non-empty question.",
                "sources": [],
                "has_sources": False,
                "error": None
            }

        if not self.vector_store_manager.is_initialized:
            return {
                "answer": "No documents have been processed yet. Please upload and process at least one PDF first.",
                "sources": [],
                "has_sources": False,
                "error": "No documents indexed."
            }

        k_val = k or self.top_k

        # 1. Retrieve most relevant document chunks
        try:
            results_with_scores = self.vector_store_manager.similarity_search_with_scores(
                query=question,
                k=k_val
            )
        except Exception as e:
            logger.error(f"Error during vector retrieval: {e}")
            return {
                "answer": "An error occurred while searching your documents.",
                "sources": [],
                "has_sources": False,
                "error": str(e)
            }

        if not results_with_scores:
            return {
                "answer": "I could not find information regarding this in the uploaded document(s).",
                "sources": [],
                "has_sources": False,
                "error": None
            }

        retrieved_docs = [doc for doc, _ in results_with_scores]

        # 2. Extract structured source citations
        sources = []
        for doc, score in results_with_scores:
            sources.append({
                "source": doc.metadata.get("source", "Unknown Document"),
                "page": doc.metadata.get("page", 1),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "content": doc.page_content,
                "score": float(score) if score is not None else 0.0
            })

        # 3. Format context string
        context_text = self._format_context(retrieved_docs)

        # 4. Check LLM availability & API key
        if not self.api_key or not config.validate_api_key(self.api_key):
            # Graceful fallback: return top retrieved sources and clear explanation
            preview_snippet = sources[0]["content"][:300] if sources else ""
            msg = (
                "⚠️ **OpenAI API Key Required for Generative Answers**\n\n"
                "The FAISS vector engine successfully retrieved matching content from your documents, "
                "but an OpenAI API key is needed to synthesize the final natural-language answer.\n\n"
                f"**Top Retrieved Match from {sources[0]['source']} (Page {sources[0]['page']}):**\n"
                f"> \"{preview_snippet}...\"\n\n"
                "👉 Please enter your OpenAI API key in the left sidebar to enable full AI answers."
            )
            return {
                "answer": msg,
                "sources": sources,
                "has_sources": True,
                "error": "Missing or invalid OpenAI API key."
            }

        # 5. Invoke LLM with grounded prompt
        try:
            llm = self._get_llm()
            messages = self._prompt_template.format_messages(
                context=context_text,
                question=question.strip()
            )
            response = llm.invoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)

            return {
                "answer": answer_text,
                "sources": sources,
                "has_sources": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"OpenAI LLM invocation failed: {e}")
            return {
                "answer": f"Unable to generate response due to an LLM service error: {str(e)}",
                "sources": sources,
                "has_sources": True,
                "error": str(e)
            }
