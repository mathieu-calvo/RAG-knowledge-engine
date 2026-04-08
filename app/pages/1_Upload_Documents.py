import os
import sys
import tempfile

import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from rag_engine.chunking.strategies import chunk_documents
from rag_engine.config import get_settings
from rag_engine.loaders import load_documents
from rag_engine.vectorstore.chroma_store import add_documents

st.title("Upload Documents")

# Initialize session state
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

settings = get_settings()

st.markdown("Upload files or provide URLs to build your knowledge base.")

# File upload
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "md", "csv", "txt"],
    accept_multiple_files=True,
)

# URL input
url_input = st.text_input("Or enter a URL to scrape:", placeholder="https://example.com/article")

# Chunk settings
col1, col2 = st.columns(2)
with col1:
    chunk_size = st.slider("Chunk size", 128, 2048, settings.chunk_size, step=64)
with col2:
    chunk_overlap = st.slider("Chunk overlap", 0, 256, settings.chunk_overlap, step=10)

if st.button("Ingest Documents", type="primary"):
    all_docs = []

    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"Loading {file.name}..."):
                # Write to temp file for loader compatibility
                suffix = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                try:
                    docs = load_documents(tmp_path)
                    all_docs.extend(docs)
                    st.success(f"Loaded {len(docs)} document(s) from {file.name}")
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")
                finally:
                    os.unlink(tmp_path)

    # Process URL
    if url_input:
        with st.spinner(f"Scraping {url_input}..."):
            try:
                docs = load_documents(url_input)
                all_docs.extend(docs)
                st.success(f"Loaded {len(docs)} document(s) from URL")
            except Exception as e:
                st.error(f"Error loading URL: {e}")

    if all_docs:
        # Chunk
        with st.spinner("Chunking documents..."):
            chunks = chunk_documents(
                all_docs, strategy="recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        # Embed and store
        with st.spinner(f"Embedding and storing {len(chunks)} chunks..."):
            try:
                add_documents(chunks)
                st.session_state.ingested_docs.extend([
                    {"name": d.metadata.get("source", "unknown"), "chunks": len(chunks)}
                    for d in all_docs
                ])
                st.success(f"Ingested {len(chunks)} chunks from {len(all_docs)} document(s)")
            except Exception as e:
                st.error(f"Error during embedding: {e}")
    elif not uploaded_files and not url_input:
        st.warning("Please upload files or enter a URL first.")

# Show ingested documents
if st.session_state.ingested_docs:
    st.markdown("### Ingested Documents")
    for doc in st.session_state.ingested_docs:
        st.markdown(f"- **{doc['name']}** ({doc['chunks']} chunks)")
