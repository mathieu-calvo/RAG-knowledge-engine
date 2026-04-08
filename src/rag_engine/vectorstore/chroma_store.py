from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_engine.config import get_settings
from rag_engine.embeddings.manager import get_embedding_model


def get_vectorstore(
    collection_name: str = "rag_documents",
    embedding_model: Embeddings | None = None,
    persist_directory: str | None = None,
) -> Chroma:
    """Create or load a ChromaDB vector store.

    Args:
        collection_name: Name of the ChromaDB collection.
        embedding_model: Embedding model to use. Defaults to the configured model.
        persist_directory: Directory for persistent storage. Defaults to config value.

    Returns:
        A LangChain Chroma vector store instance.
    """
    settings = get_settings()
    persist_directory = persist_directory or settings.chroma_persist_dir
    embedding_model = embedding_model or get_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )


def add_documents(
    documents: list[Document],
    collection_name: str = "rag_documents",
    embedding_model: Embeddings | None = None,
    persist_directory: str | None = None,
) -> Chroma:
    """Add documents to a ChromaDB vector store.

    Creates the collection if it doesn't exist, or adds to an existing one.

    Args:
        documents: List of Document objects to add.
        collection_name: Name of the ChromaDB collection.
        embedding_model: Embedding model to use.
        persist_directory: Directory for persistent storage.

    Returns:
        The Chroma vector store with the added documents.
    """
    vectorstore = get_vectorstore(collection_name, embedding_model, persist_directory)
    vectorstore.add_documents(documents)
    return vectorstore


def clear_vectorstore(
    collection_name: str = "rag_documents",
    persist_directory: str | None = None,
) -> None:
    """Delete all documents from a ChromaDB collection.

    Args:
        collection_name: Name of the collection to clear.
        persist_directory: Directory of the persistent storage.
    """
    import chromadb

    settings = get_settings()
    persist_directory = persist_directory or settings.chroma_persist_dir

    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass  # Collection doesn't exist, nothing to clear
