import streamlit as st
from pathlib import Path

from src.retrieval import get_reranked_retriever, get_hybrid_retriever, get_router
from src.chain import get_rag_chain, get_query_rewriter, get_faithfulness_checker, answer_question
from src.ingestion import load_documents, split_documents, add_to_vectorstore, get_vectorstore, get_indexed_files, mark_files_indexed

# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Doc RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="margin-bottom: 0.2rem;">📄 Multi-Doc RAG Assistant</h1>
        <p style="color: #9CA3AF; font-size: 0.95rem;">
            Ask questions across multiple documents, grounded and cited.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .source-card {
        background-color: #1A1D24;
        border: 1px solid #2D3139;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .source-card .source-label {
        color: #6366F1;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }
    .source-card .source-text {
        color: #9CA3AF;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    .strategy-badge {
        display: inline-block;
        background-color: #2D3139;
        color: #9CA3AF;
        font-size: 0.75rem;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Data directory
# --------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Sidebar - Document Upload
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding-bottom: 0.5rem;">
            <h3 style="margin-bottom: 0;">📁 Documents</h3>
            <p style="color: #9CA3AF; font-size: 0.85rem; margin-top: 0.2rem;">
                Upload files to build your knowledge base
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or CSV files",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"**{len(uploaded_files)} file(s) selected**")
        for f in uploaded_files:
            st.markdown(f"&nbsp;&nbsp;📄 {f.name}")
    else:
        st.info("Upload one or more documents to get started.")

    st.divider()

# --------------------------------------------------
# Index Documents
# --------------------------------------------------

if uploaded_files and st.sidebar.button("Index documents"):
    already_indexed = get_indexed_files()
    new_files = [f for f in uploaded_files if f.name not in already_indexed]

    if not new_files:
        st.sidebar.info("All selected files are already indexed.")
    else:
        # Save files to data/
        for f in new_files:
            save_path = DATA_DIR / f.name

            with open(save_path, "wb") as out:
                out.write(f.getbuffer())

        # Process documents
        with st.spinner(f"Indexing {len(new_files)} new file(s)..."):
            docs = load_documents(str(DATA_DIR), only_files={f.name for f in new_files})
            chunks = split_documents(docs)
            add_to_vectorstore(chunks)
            mark_files_indexed([f.name for f in new_files])

        # Clear cached resources so the new
        # vectorstore is loaded
        st.cache_resource.clear()
        st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(new_files)} new file(s).")

# --------------------------------------------------
# Load RAG Pipeline
# --------------------------------------------------

@st.cache_resource
def load_pipeline():
    vectorstore = get_vectorstore()

    if vectorstore is None:
        return None

    return {
        "vectorstore": vectorstore,
        "hybrid_retriever": get_hybrid_retriever(vectorstore),
        "router": get_router(),
        "rag_chain": get_rag_chain(),
        "rewriter": get_query_rewriter(),
        "checker": get_faithfulness_checker(),
    }

# --------------------------------------------------
# Chat UI
# --------------------------------------------------

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

AVATARS = {"user": "🧑", "assistant": "🤖"}

for role, text in st.session_state.messages:
    with st.chat_message(role, avatar=AVATARS[role]):
        st.write(text)

# --------------------------------------------------
# User Input
# --------------------------------------------------

user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Store user message
    st.session_state.messages.append(("user", user_input))
    
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.write(user_input)

    # Load pipeline
    pipeline = load_pipeline()

    # --------------------------------------------------
    # No documents indexed
    # --------------------------------------------------

    if pipeline is None:
        assistant_text = "Please upload and index at least one document first."

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.write(assistant_text)

        st.session_state.messages.append(("assistant", assistant_text))

    # --------------------------------------------------
    # Documents available
    # --------------------------------------------------

    else:
        try:
            with st.spinner("Thinking..."):
                result = answer_question(
                    user_input,
                    st.session_state.messages[:-1],
                    pipeline["rag_chain"],
                    pipeline["rewriter"],
                    pipeline["checker"],
                    pipeline["vectorstore"],
                    pipeline["router"],
                    pipeline["hybrid_retriever"],
                )

            # Save assistant answer
            st.session_state.messages.append(("assistant", result["answer"]))

            # Display answer
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                st.write(result["answer"])

                # Faithfulness warning
                if not result["is_faithful"]:
                    st.warning(f"⚠️ Possible hallucination: {result['faithfulness_reason']}")

                # --------------------------------------------------
                # Sources
                # --------------------------------------------------

    
                sst.markdown(f'<span class="strategy-badge">🔍 {result["strategy"]}</span>', unsafe_allow_html=True)

                with st.expander("📚 Sources"):
                      for doc in result["sources"]:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page")
                        label = f"{source}, page {page + 1}" if page is not None else source
                        snippet = doc.page_content[:200].replace("\n", " ") + "..."
                        st.markdown(f"""<div class="source-card"><div class="source-label">📄 {label}</div><div class="source-text">{snippet}</div></div>
                        """   ,                       
                        unsafe_allow_html=True,
                    )

        # --------------------------------------------------
        # Error handling
        # --------------------------------------------------

        except Exception as e:
            error_text = str(e)

            if "rate limit" in error_text.lower() or "429" in error_text:
                friendly = (
                    "I'm getting a lot of requests right now (free-tier rate limit). "
                    "Please wait about a minute and try again."
                )
            else:
                friendly = (
                    "Something went wrong while generating the answer. "
                    "Please try again."
                )

            st.session_state.messages.append(("assistant", friendly))

            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                st.write("Please upload and index at least one document first.")