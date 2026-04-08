import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents():
    """Return a list of sample Document objects for testing."""
    return [
        Document(
            page_content="Retrieval-Augmented Generation (RAG) combines retrieval with generation.",
            metadata={"source": "test_doc.pdf", "page": 0},
        ),
        Document(
            page_content="Vector databases store embeddings for fast similarity search.",
            metadata={"source": "test_doc.pdf", "page": 1},
        ),
        Document(
            page_content="LangChain provides abstractions for building LLM applications.",
            metadata={"source": "test_doc.md", "page": 0},
        ),
    ]


@pytest.fixture
def sample_texts():
    """Return sample text strings for chunking and embedding tests."""
    return [
        "The transformer architecture was introduced in the paper 'Attention Is All You Need'.",
        "Self-attention allows the model to weigh the importance of different input tokens.",
        "BERT and GPT are both based on the transformer architecture.",
    ]
