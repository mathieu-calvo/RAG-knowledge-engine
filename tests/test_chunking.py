import pytest
from langchain_core.documents import Document

from rag_engine.chunking.strategies import chunk_documents, get_text_splitter


class TestGetTextSplitter:
    def test_recursive_strategy(self):
        splitter = get_text_splitter("recursive", chunk_size=100, chunk_overlap=20)
        assert splitter is not None

    def test_character_strategy(self):
        splitter = get_text_splitter("character", chunk_size=100, chunk_overlap=20)
        assert splitter is not None

    def test_token_strategy(self):
        splitter = get_text_splitter("token", chunk_size=100, chunk_overlap=20)
        assert splitter is not None

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_text_splitter("unknown")


class TestChunkDocuments:
    def test_chunk_documents_basic(self):
        docs = [
            Document(
                page_content="This is a long document. " * 50,
                metadata={"source": "test.txt"},
            )
        ]
        chunks = chunk_documents(docs, strategy="recursive", chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(isinstance(c, Document) for c in chunks)
        # Metadata should be preserved
        assert all(c.metadata["source"] == "test.txt" for c in chunks)

    def test_chunk_size_respected(self):
        docs = [
            Document(
                page_content="Word " * 200,
                metadata={"source": "test.txt"},
            )
        ]
        chunk_size = 100
        chunks = chunk_documents(docs, strategy="recursive", chunk_size=chunk_size, chunk_overlap=0)
        # Most chunks should be near chunk_size (some may slightly exceed)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size * 1.5  # Allow some tolerance

    def test_short_document_not_split(self):
        docs = [
            Document(page_content="Short text.", metadata={"source": "test.txt"})
        ]
        chunks = chunk_documents(docs, strategy="recursive", chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_multiple_documents(self):
        docs = [
            Document(page_content="Document one. " * 50, metadata={"source": "a.txt"}),
            Document(page_content="Document two. " * 50, metadata={"source": "b.txt"}),
        ]
        chunks = chunk_documents(docs, strategy="recursive", chunk_size=100, chunk_overlap=10)
        sources = {c.metadata["source"] for c in chunks}
        assert "a.txt" in sources
        assert "b.txt" in sources
