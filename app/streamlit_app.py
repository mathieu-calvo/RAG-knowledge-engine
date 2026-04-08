import os
import sys

import streamlit as st

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

st.set_page_config(
    page_title="RAG Knowledge Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("RAG Knowledge Engine")
st.sidebar.markdown(
    "An educational RAG pipeline built with LangChain, ChromaDB, "
    "and swappable LLM providers."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pages:**\n"
    "- **Upload Documents** — ingest your data\n"
    "- **Chat** — ask questions\n"
    "- **Settings** — configure the pipeline"
)

st.title("RAG Knowledge Engine")
st.markdown(
    """
    Welcome to the RAG Knowledge Engine — a **Retrieval-Augmented Generation** demo
    that lets you:

    1. **Upload documents** (PDF, Markdown, CSV) or provide URLs
    2. **Ask questions** and get answers grounded in your documents
    3. **Configure** the LLM provider, chunk size, retrieval strategy, and more

    Use the sidebar to navigate between pages.

    ### Quick Start
    1. Go to **Upload Documents** and add some files
    2. Go to **Chat** and start asking questions
    3. Adjust settings in **Settings** to experiment with different configurations
    """
)
