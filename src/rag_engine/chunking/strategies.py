from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from rag_engine.config import get_settings

STRATEGY_MAP = {
    "character": CharacterTextSplitter,
    "recursive": RecursiveCharacterTextSplitter,
    "token": TokenTextSplitter,
}


def get_text_splitter(
    strategy: str = "recursive",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    **kwargs,
):
    """Create a text splitter with the specified strategy.

    Args:
        strategy: One of "character", "recursive", "token".
        chunk_size: Maximum size of each chunk. Defaults to config value.
        chunk_overlap: Overlap between consecutive chunks. Defaults to config value.
        **kwargs: Additional arguments passed to the splitter constructor.

    Returns:
        A LangChain TextSplitter instance.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if strategy not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGY_MAP)}")

    splitter_class = STRATEGY_MAP[strategy]
    return splitter_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )


def chunk_documents(
    documents: list[Document],
    strategy: str = "recursive",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    **kwargs,
) -> list[Document]:
    """Split a list of Documents into smaller chunks.

    Args:
        documents: List of Document objects to split.
        strategy: Chunking strategy ("character", "recursive", "token").
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.
        **kwargs: Additional arguments for the splitter.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = get_text_splitter(strategy, chunk_size, chunk_overlap, **kwargs)
    return splitter.split_documents(documents)
