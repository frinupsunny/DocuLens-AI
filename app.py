"""
DocuLens AI - Intelligent PDF Question Answering System
Frontend Application using Streamlit, FAISS, and LangChain RAG.
"""

import io
import streamlit as st
import config
from src.pdf_processor import PDFProcessor, PDFProcessingError
from src.text_chunker import TextChunker
from src.embeddings import get_embedding_model
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.chat_manager import ChatManager

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="DocuLens AI - Intelligent PDF QA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Custom CSS for SaaS Dashboard Aesthetic
# ==============================================================================
st.markdown("""
<style>
    /* Google Fonts & Base Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global Page Background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header & Titles */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* Modern Card Container */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    
    .feature-heading {
        font-size: 1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.25rem;
    }
    
    .feature-text {
        font-size: 0.875rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Metric Card */
    .stat-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stat-badge-ready {
        background-color: #DEF7EC;
        color: #03543F;
    }
    .stat-badge-idle {
        background-color: #FEF08A;
        color: #713F12;
    }

    /* Source Citation Card */
    .source-box {
        background-color: #FFFFFF;
        border-left: 4px solid #6366F1;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }
    
    .source-badge {
        background-color: #EEF2FF;
        color: #4F46E5;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        margin-right: 0.5rem;
    }

    .source-excerpt {
        font-size: 0.85rem;
        color: #334155;
        font-style: italic;
        margin-top: 0.4rem;
        line-height: 1.45;
    }

    /* Streamlit Chat tweaks */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Primary Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# State Initialization
# ==============================================================================
def init_session_state():
    """Initialize persistent session components."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()

    if "embeddings" not in st.session_state:
        # Load local Hugging Face embeddings singleton
        with st.spinner("Initializing Hugging Face embedding model (all-MiniLM-L6-v2)..."):
            st.session_state.embeddings = get_embedding_model()

    if "vector_manager" not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager(st.session_state.embeddings)

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "total_pages" not in st.session_state:
        st.session_state.total_pages = 0

    if "prompt_input_override" not in st.session_state:
        st.session_state.prompt_input_override = None


init_session_state()


# ==============================================================================
# Helper Functions
# ==============================================================================
def format_file_size(size_bytes: int) -> str:
    """Format bytes into readable KB or MB."""
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


# ==============================================================================
# Sidebar: Control Panel & Document Management
# ==============================================================================
with st.sidebar:
    # Branding Header
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="background: linear-gradient(135deg, #4F46E5, #9333EA); width: 38px; height: 38px; border-radius: 9px; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px; font-weight: bold;">
                🔍
            </div>
            <div>
                <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #0F172A;">DocuLens AI</h2>
                <p style="margin: 0; font-size: 0.75rem; color: #64748B;">Intelligent Document Intelligence</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Section 1: LLM & API Configuration ---
    st.markdown("#### ⚙️ Configuration")

    # API Key Handling: Environment variable or User Input
    env_key = config.OPENAI_API_KEY
    user_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=env_key if env_key else "",
        placeholder="sk-proj-...",
        help="Stored only in memory during your active browser session."
    )

    is_key_valid = config.validate_api_key(user_api_key)
    if is_key_valid:
        st.markdown(
            '<span class="stat-badge stat-badge-ready">✓ API Key Active</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="stat-badge stat-badge-idle">⚠️ API Key Required for Generative Answers</span>',
            unsafe_allow_html=True
        )

    # Advanced Settings Expander
    with st.expander("Model & Retrieval Parameters", expanded=False):
        model_choice = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o-mini offers exceptional speed and accuracy for document RAG."
        )
        top_k = st.slider(
            "Retrieved Chunks (Top-K)",
            min_value=1,
            max_value=8,
            value=config.DEFAULT_TOP_K,
            help="Number of relevant document excerpts provided to the LLM."
        )

    st.divider()

    # --- Section 2: PDF Upload & Ingestion ---
    st.markdown("#### 📄 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple research papers, reports, or business documents."
    )

    if uploaded_files:
        st.markdown(f"**Selected Files ({len(uploaded_files)}):**")
        for f in uploaded_files:
            is_indexed = f.name in st.session_state.processed_files
            status_badge = (
                '<span style="color: #059669; font-weight: 600; font-size: 0.75rem;">✓ Indexed</span>'
                if is_indexed else
                '<span style="color: #D97706; font-size: 0.75rem;">Pending</span>'
            )
            st.markdown(
                f"<div style='font-size: 0.85rem; display: flex; justify-content: space-between; padding: 2px 0;'>"
                f"<span>📄 {f.name} ({format_file_size(f.size)})</span>"
                f"{status_badge}"
                f"</div>",
                unsafe_allow_html=True
            )

        # Process Documents Button
        process_clicked = st.button("⚡ Process Documents", type="primary", use_container_width=True)

        if process_clicked:
            # Check for new files that have not been indexed yet
            new_files_to_process = [
                {"file": f, "name": f.name, "size": f.size}
                for f in uploaded_files
                if f.name not in st.session_state.processed_files
            ]

            if not new_files_to_process:
                st.info("All uploaded documents are already processed and indexed.")
            else:
                progress_bar = st.progress(0, text="Extracting and preprocessing text...")
                processor = PDFProcessor()
                chunker = TextChunker()

                try:
                    # 1. Extract documents
                    extraction_result = processor.process_multiple_pdfs(new_files_to_process)
                    docs = extraction_result["documents"]
                    progress_bar.progress(40, text="Splitting text into overlapping semantic chunks...")

                    if not docs:
                        st.error("No extractable text found in the provided PDF(s).")
                    else:
                        # 2. Chunk text
                        chunks = chunker.split_documents(docs)
                        progress_bar.progress(70, text="Generating embeddings and indexing in FAISS...")

                        # 3. Add to vector store
                        st.session_state.vector_manager.add_documents(chunks)

                        # Update session tracker
                        for fname in extraction_result["processed_files"]:
                            st.session_state.processed_files.add(fname)
                        st.session_state.total_pages += extraction_result["total_pages"]

                        progress_bar.progress(100, text="Indexing complete!")
                        st.success(
                            f"Successfully indexed {len(extraction_result['processed_files'])} "
                            f"document(s) ({len(chunks)} chunks)!"
                        )

                        if extraction_result["failed_files"]:
                            for fail_name, err in extraction_result["failed_files"].items():
                                st.warning(f"Could not process '{fail_name}': {err}")

                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

    st.divider()

    # --- Section 3: Index Statistics ---
    st.markdown("#### 📊 System Metrics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Documents", len(st.session_state.processed_files))
        st.metric("Total Pages", st.session_state.total_pages)
    with col_stat2:
        st.metric("Total Chunks", st.session_state.vector_manager.total_chunks_indexed)
        status_label = "Ready" if st.session_state.vector_manager.is_initialized else "Awaiting Data"
        st.metric("Index Status", status_label)

    st.divider()

    # --- Section 4: Session Controls ---
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_manager.clear()
            st.rerun()

    with col_btn2:
        if st.button("Reset All", use_container_width=True):
            st.session_state.vector_manager.clear()
            st.session_state.chat_manager.clear()
            st.session_state.processed_files.clear()
            st.session_state.total_pages = 0
            st.rerun()


# ==============================================================================
# Main Dashboard
# ==============================================================================
st.markdown('<div class="hero-title">Your Documents. Your Knowledge.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Upload PDFs, ask questions, and receive accurate, grounded answers backed by source citations.</div>', unsafe_allow_html=True)

# Instantiate RAG pipeline with active settings
rag_pipeline = RAGPipeline(
    vector_store_manager=st.session_state.vector_manager,
    api_key=user_api_key,
    model_name=model_choice,
    top_k=top_k
)

# Landing State: Displayed when no documents are indexed yet
if not st.session_state.vector_manager.is_initialized:
    st.markdown("""
        <div style="background: white; border-radius: 12px; border: 1px solid #E2E8F0; padding: 24px; margin-bottom: 24px;">
            <h3 style="color: #0F172A; margin-top: 0;">Welcome to DocuLens AI</h3>
            <p style="color: #475569; font-size: 0.95rem;">
                DocuLens AI combines <strong>local Hugging Face embeddings</strong>, high-performance <strong>FAISS vector search</strong>, 
                and <strong>OpenAI Large Language Models</strong> to provide accurate, hallucination-resistant answers grounded strictly in your PDFs.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">📑</span>
                <div class="feature-heading">1. Ingest Documents</div>
                <div class="feature-text">Upload single or multi-page PDFs in the sidebar. Text is extracted, sanitized, and tagged with exact page numbers.</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">⚡</span>
                <div class="feature-heading">2. Local Vector Indexing</div>
                <div class="feature-text">Documents are split into semantic chunks and embedded locally using Hugging Face <code>all-MiniLM-L6-v2</code> without external token costs.</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">🎯</span>
                <div class="feature-heading">3. Grounded Q&A</div>
                <div class="feature-text">Ask complex questions. The LLM synthesizes answers strictly based on retrieved context and provides verifiable citations.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 💡 Example Questions You Can Ask Once Documents Are Uploaded:")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("- *\"What is the executive summary of this report?\"*")
        st.markdown("- *\"What key methodologies and findings are discussed on page 3?\"*")
    with ex_col2:
        st.markdown("- *\"Summarize the conclusions and recommended next steps.\"*")
        st.markdown("- *\"Compare the data presented across the uploaded documents.\"*")

    st.info("👈 Upload your PDF files in the sidebar and click **Process Documents** to begin exploring!")

else:
    # Quick Suggested Questions Chips
    st.markdown("<p style='font-size: 0.85rem; color: #64748B; margin-bottom: 6px;'>Suggested queries:</p>", unsafe_allow_html=True)
    suggest_col1, suggest_col2, suggest_col3 = st.columns(3)
    with suggest_col1:
        if st.button("📋 Summarize key findings", use_container_width=True):
            st.session_state.prompt_input_override = "Summarize the primary objectives and key findings of the document(s)."
    with suggest_col2:
        if st.button("🔍 What are the conclusions?", use_container_width=True):
            st.session_state.prompt_input_override = "What are the main conclusions and recommendations provided?"
    with suggest_col3:
        if st.button("📊 Extract main metrics", use_container_width=True):
            st.session_state.prompt_input_override = "Extract and list the main quantitative metrics or statistics mentioned."

# ==============================================================================
# Chat Display
# ==============================================================================
# Render chat history
for message in st.session_state.chat_manager.messages:
    if message.role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
            if message.sources:
                with st.expander(f"📚 View Retrieved Sources ({len(message.sources)})", expanded=False):
                    for idx, src in enumerate(message.sources, 1):
                        st.markdown(f"""
                            <div class="source-box">
                                <div>
                                    <span class="source-badge">Source {idx}</span>
                                    <strong>{src['source']}</strong> (Page {src['page']})
                                </div>
                                <div class="source-excerpt">
                                    "{src['content'][:350]}..."
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

# ==============================================================================
# User Input Processing
# ==============================================================================
prompt = st.chat_input("Ask a question about your uploaded documents...")

# Handle suggested button click override
if st.session_state.prompt_input_override:
    prompt = st.session_state.prompt_input_override
    st.session_state.prompt_input_override = None

if prompt:
    # 1. Add user message
    st.session_state.chat_manager.add_user_message(prompt)
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Generate answer
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching document context and generating answer..."):
            response = rag_pipeline.answer_question(prompt)
            answer_text = response["answer"]
            sources = response["sources"]

            st.markdown(answer_text)

            if sources:
                with st.expander(f"📚 View Retrieved Sources ({len(sources)})", expanded=False):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"""
                            <div class="source-box">
                                <div>
                                    <span class="source-badge">Source {idx}</span>
                                    <strong>{src['source']}</strong> (Page {src['page']})
                                </div>
                                <div class="source-excerpt">
                                    "{src['content'][:350]}..."
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

    # 3. Save assistant message to history
    st.session_state.chat_manager.add_assistant_message(
        content=answer_text,
        sources=sources
    )
