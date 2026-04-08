import os
import sys

import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from rag_engine.config import get_settings
from rag_engine.vectorstore.chroma_store import clear_vectorstore

st.title("Settings")

settings = get_settings()

st.markdown("### LLM Configuration")

col1, col2 = st.columns(2)
with col1:
    provider = st.selectbox(
        "LLM Provider",
        ["openai", "anthropic"],
        index=0 if settings.llm_provider == "openai" else 1,
    )

with col2:
    if provider == "openai":
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    else:
        model_options = ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]

    model = st.selectbox("Model", model_options)

temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)

st.markdown("### Embedding Configuration")
embedding_model = st.selectbox(
    "Embedding Model",
    ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
)

st.markdown("### RAG Configuration")

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.slider("Chunk Size", 128, 2048, settings.chunk_size, step=64)
with col2:
    chunk_overlap = st.slider("Chunk Overlap", 0, 256, settings.chunk_overlap, step=10)

retrieval_strategy = st.selectbox("Retrieval Strategy", ["similarity", "mmr"], index=1)
top_k = st.slider("Top K Results", 1, 10, settings.retrieval_top_k)

st.markdown("---")

st.markdown("### Danger Zone")

if st.button("Clear Vector Store", type="secondary"):
    try:
        clear_vectorstore()
        if "ingested_docs" in st.session_state:
            st.session_state.ingested_docs = []
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.success("Vector store cleared. All documents have been removed.")
    except Exception as e:
        st.error(f"Error clearing vector store: {e}")

st.markdown("---")
st.markdown(
    "*Note: Settings changes take effect on the next query. "
    "To apply new chunk size/overlap, re-upload your documents.*"
)
