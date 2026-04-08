import os
import sys

import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from rag_engine.chains.rag_chain import build_rag_chain_with_sources
from rag_engine.config import get_settings
from rag_engine.llm.provider import LLMProvider
from rag_engine.retrieval.retriever import RetrieverFactory
from rag_engine.vectorstore.chroma_store import get_vectorstore

st.title("Chat with Your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

settings = get_settings()

# Sidebar settings
st.sidebar.markdown("### Retrieval Settings")
strategy = st.sidebar.selectbox("Retrieval strategy", ["similarity", "mmr"], index=1)
top_k = st.sidebar.slider("Number of chunks to retrieve", 1, 10, settings.retrieval_top_k)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**[{i}] {source['name']}**")
                    st.text(source["content"][:200] + "...")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                vectorstore = get_vectorstore()
                retriever = RetrieverFactory.create_retriever(
                    vectorstore, strategy=strategy, top_k=top_k
                )
                llm = LLMProvider.get_llm(temperature=0.0)
                chain = build_rag_chain_with_sources(retriever, llm)

                result = chain.invoke({"question": prompt})

                answer = result["answer"]
                source_docs = result["source_documents"]

                st.markdown(answer)

                sources = [
                    {"name": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
                    for doc in source_docs
                ]

                if sources:
                    with st.expander(f"Sources ({len(sources)} documents)"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**[{i}] {source['name']}**")
                            st.text(source["content"][:200] + "...")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                error_msg = f"Error: {e}. Make sure you've uploaded documents first."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
