import tempfile

import pytest
from langchain_core.documents import Document

from rag_engine.vectorstore.chroma_store import add_documents, clear_vectorstore, get_vectorstore


class FakeEmbeddings:
    """Deterministic fake embeddings for testing (no API calls)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10) / 10.0] * 384 for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10) / 10.0] * 384


class TestVectorStore:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "chroma_test")

    @pytest.fixture
    def fake_embeddings(self):
        return FakeEmbeddings()

    def test_get_vectorstore(self, temp_dir, fake_embeddings):
        vs = get_vectorstore(
            collection_name="test",
            embedding_model=fake_embeddings,
            persist_directory=temp_dir,
        )
        assert vs is not None

    def test_add_and_retrieve_documents(self, temp_dir, fake_embeddings):
        docs = [
            Document(page_content="RAG combines retrieval with generation.", metadata={"source": "a"}),
            Document(page_content="Vector databases enable fast similarity search.", metadata={"source": "b"}),
        ]

        vs = add_documents(
            docs,
            collection_name="test_add",
            embedding_model=fake_embeddings,
            persist_directory=temp_dir,
        )

        results = vs.similarity_search("retrieval generation", k=2)
        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)

    def test_clear_vectorstore(self, temp_dir, fake_embeddings):
        docs = [Document(page_content="Test document.", metadata={"source": "test"})]
        add_documents(
            docs,
            collection_name="test_clear",
            embedding_model=fake_embeddings,
            persist_directory=temp_dir,
        )

        # Should not raise
        clear_vectorstore(collection_name="test_clear", persist_directory=temp_dir)

    def test_clear_nonexistent_collection(self, temp_dir):
        # Should not raise even if collection doesn't exist
        clear_vectorstore(collection_name="nonexistent", persist_directory=temp_dir)
