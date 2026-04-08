import pytest
from langchain_core.documents import Document

from rag_engine.retrieval.retriever import RetrieverFactory


# Reuse the FakeEmbeddings from test_vectorstore
class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i) / len(texts)] * 384 for i, t in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * 384


def _make_vectorstore(tmp_path):
    """Create a small Chroma vectorstore for testing."""
    from langchain_chroma import Chroma

    docs = [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "a"}),
        Document(page_content="Vector databases store embeddings.", metadata={"source": "b"}),
        Document(page_content="LangChain is a framework for LLM apps.", metadata={"source": "c"}),
        Document(page_content="Chunking splits documents into pieces.", metadata={"source": "d"}),
    ]
    return Chroma.from_documents(
        docs,
        embedding=FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test",
    )


class TestRetrieverFactory:
    def test_similarity_retriever(self, tmp_path):
        vs = _make_vectorstore(tmp_path)
        retriever = RetrieverFactory.create_retriever(vs, strategy="similarity", top_k=2)
        results = retriever.invoke("What is RAG?")
        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)

    def test_mmr_retriever(self, tmp_path):
        vs = _make_vectorstore(tmp_path)
        retriever = RetrieverFactory.create_retriever(vs, strategy="mmr", top_k=2)
        results = retriever.invoke("What is RAG?")
        assert len(results) == 2

    def test_unknown_strategy(self, tmp_path):
        vs = _make_vectorstore(tmp_path)
        with pytest.raises(ValueError, match="Unknown strategy"):
            RetrieverFactory.create_retriever(vs, strategy="unknown")
