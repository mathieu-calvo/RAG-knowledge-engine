# RAG Knowledge Engine

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange.svg)](https://www.trychroma.com/)

A **production-quality Retrieval-Augmented Generation (RAG) pipeline** with educational notebooks — built with LangChain, ChromaDB, and swappable LLM providers (Google Gemini, OpenAI, Anthropic).

This project is both a **working RAG system** and a **step-by-step tutorial** explaining how RAG is built in practice.

**Runs for free** — defaults to Google Gemini (free tier) and local HuggingFace embeddings. No paid API required.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────────┐
│  Documents   │───>│   Chunking   │───>│ Embeddings │───>│ Vector Store│
│ (PDF,MD,CSV) │    │  & Splitting │    │   Model    │    │  (ChromaDB) │
└─────────────┘    └──────────────┘    └────────────┘    └──────┬──────┘
                                                                │
                                                                │ retrieve
                                                                │
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────┴────────┐
│   Answer     │<──│     LLM      │<──│   Prompt    │<──│  Retriever   │
│              │    │(Gemini/GPT/   │   │ (question + │    │ (top-k docs) │
│              │    │    Claude)    │
└─────────────┘    └──────────────┘    │  context)   │    └──────────────┘
                                       └────────────┘
```

## Features

- [x] **Multi-format document loading** — PDF, Markdown, CSV, web pages
- [x] **Configurable chunking** — character, recursive, and token-based strategies
- [x] **Semantic search** with ChromaDB vector store
- [x] **Multiple retrieval strategies** — similarity search, MMR, score threshold
- [x] **Swappable LLM providers** — Google Gemini (free), OpenAI, and Anthropic via a single factory pattern
- [x] **RAG evaluation** — built-in evaluation pipeline with RAGAS support
- [x] **Advanced techniques** — HyDE, multi-query retrieval, cross-encoder re-ranking
- [x] **Interactive Streamlit app** — upload docs, chat, and configure settings
- [x] **9 educational notebooks** — from "What is RAG?" to advanced retrieval
- [x] **Clean Python package** — modular, tested, well-documented source code

## Built With

- **[LangChain](https://python.langchain.com/)** — LLM application framework
- **[ChromaDB](https://www.trychroma.com/)** — Local vector database
- **[Google Gemini](https://ai.google.dev/)** — Free-tier LLM (default)
- **[HuggingFace](https://huggingface.co/)** — Local embeddings (default, free)
- **[OpenAI](https://platform.openai.com/)** — GPT models and embeddings (optional, paid)
- **[Anthropic](https://www.anthropic.com/)** — Claude models (optional, paid)
- **[Streamlit](https://streamlit.io/)** — Interactive web app
- **[RAGAS](https://docs.ragas.io/)** — RAG evaluation framework

## Getting Started

### Prerequisites

- Python 3.10+
- A Google API key (**free** — see step 2 below)
- OpenAI or Anthropic API keys are **optional** (paid) if you want to use those providers

### Installation

**1. Clone and install:**

```bash
git clone https://github.com/mathieu-calvo/RAG-knowledge-engine-.git
cd RAG-knowledge-engine-
pip install -e ".[dev,app,eval]"
```

**2. Get a free Google API key:**

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (it starts with `AIza...`)

**3. Create your `.env` configuration file:**

The project reads secrets and settings from a file called `.env` in the project root. This file is **not** committed to Git (it's in `.gitignore`) so your API keys stay private.

A template called `.env.example` is provided. Copy it to create your own `.env`:

```bash
cp .env.example .env
```

Then open `.env` in any text editor and paste your Google API key:

```
GOOGLE_API_KEY=AIzaSy...your-actual-key-here
```

That's it — the rest of the defaults (Gemini Flash for LLM, HuggingFace for embeddings) are already set and work for free.

> **Want to use OpenAI or Anthropic instead?** Install the optional dependencies and update `.env`:
> ```bash
> pip install -e ".[openai]"    # for GPT models
> pip install -e ".[anthropic]" # for Claude models
> ```
> Then set `LLM_PROVIDER=openai` (or `anthropic`) and the corresponding API key in `.env`.

### Quick Start

```python
from rag_engine.loaders import load_documents
from rag_engine.chunking.strategies import chunk_documents
from rag_engine.vectorstore.chroma_store import add_documents
from rag_engine.retrieval.retriever import RetrieverFactory
from rag_engine.llm.provider import LLMProvider
from rag_engine.chains.rag_chain import build_rag_chain

# 1. Ingest documents
docs = load_documents("data/sample/rag_survey.md")
chunks = chunk_documents(docs, chunk_size=512)
vectorstore = add_documents(chunks)

# 2. Build the RAG chain
retriever = RetrieverFactory.create_retriever(vectorstore, strategy="mmr")
llm = LLMProvider.get_llm()  # Uses config from .env
chain = build_rag_chain(retriever, llm)

# 3. Ask questions
answer = chain.invoke("What is RAG and what problem does it solve?")
print(answer)
```

## Notebook Walkthrough

| # | Notebook | Description |
|---|---------|-------------|
| 01 | [What is RAG?](notebooks/01_what_is_rag.ipynb) | RAG overview, LangChain introduction, hallucination demo |
| 02 | [Document Loading](notebooks/02_document_loading.ipynb) | Loading PDFs, Markdown, CSV, and web pages |
| 03 | [Chunking Strategies](notebooks/03_chunking_strategies.ipynb) | Comparing splitters, chunk size experiments with visualizations |
| 04 | [Embeddings & Vector Stores](notebooks/04_embeddings_and_vectorstores.ipynb) | Embedding models, ChromaDB, t-SNE visualization |
| 05 | [Retrieval Strategies](notebooks/05_retrieval_strategies.ipynb) | Similarity search vs. MMR, side-by-side comparison |
| 06 | [Building the RAG Chain](notebooks/06_rag_chain.ipynb) | LCEL composition, prompt engineering, source citations |
| 07 | [Swappable LLM Providers](notebooks/07_swappable_llm_providers.ipynb) | OpenAI vs. Anthropic, provider factory pattern |
| 08 | [Evaluation](notebooks/08_evaluation.ipynb) | RAGAS metrics, evaluation dataset, result visualization |
| 09 | [Advanced Techniques](notebooks/09_advanced_techniques.ipynb) | HyDE, multi-query retrieval, cross-encoder re-ranking |

## Streamlit App

```bash
make app
# or: streamlit run app/streamlit_app.py
```

The app has three pages:
- **Upload Documents** — drag and drop files or enter URLs
- **Chat** — conversational RAG with source citations
- **Settings** — configure LLM provider, chunk size, retrieval strategy

## Project Structure

```
RAG-knowledge-engine/
├── notebooks/              # 9 educational Jupyter notebooks
├── src/rag_engine/         # Python package
│   ├── config.py           # Centralized settings (pydantic-settings)
│   ├── loaders/            # PDF, Markdown, CSV, web loaders
│   ├── chunking/           # Text splitting strategies
│   ├── embeddings/         # Embedding model manager
│   ├── vectorstore/        # ChromaDB wrapper
│   ├── retrieval/          # Retriever factory (similarity, MMR, etc.)
│   ├── chains/             # RAG chain builder + prompt templates
│   ├── llm/                # Swappable LLM provider factory
│   └── evaluation/         # RAGAS evaluation pipeline
├── app/                    # Streamlit application
├── data/sample/            # Sample documents for demos
├── tests/                  # Pytest test suite
└── docs/                   # Architecture documentation
```

## Running Tests

```bash
# Run tests (no API keys needed)
make test

# Run all tests including API-dependent ones
make test-all
```

## Code Quality

```bash
make lint     # Run ruff linter
make format   # Auto-format with ruff
```

## Roadmap

- [ ] Async retrieval for faster response times
- [ ] Streaming LLM responses in the Streamlit app
- [ ] Additional vector database backends (FAISS, Pinecone)
- [ ] PDF table extraction
- [ ] RAG for financial documents (SEC filings, earnings reports)
- [ ] Conversation memory with persistent storage

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Mathieu Calvo — [GitHub](https://github.com/mathieu-calvo)

## References

- Lewis, P., et al. (2020). ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401)
- Gao, Y., et al. (2024). ["Retrieval-Augmented Generation for Large Language Models: A Survey"](https://arxiv.org/abs/2312.10997)
- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAGAS Documentation](https://docs.ragas.io/)
