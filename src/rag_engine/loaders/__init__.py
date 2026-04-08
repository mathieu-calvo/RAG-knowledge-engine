from pathlib import Path

from langchain_core.documents import Document

from rag_engine.loaders.csv_loader import load_csv
from rag_engine.loaders.markdown_loader import load_markdown
from rag_engine.loaders.pdf_loader import load_pdf
from rag_engine.loaders.web_loader import load_web

EXTENSION_MAP = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".csv": load_csv,
}


def load_documents(source: str, source_type: str | None = None, **kwargs) -> list[Document]:
    """Load documents from a file path or URL.

    Automatically detects the source type from the file extension,
    or you can specify it explicitly.

    Args:
        source: File path or URL.
        source_type: One of "pdf", "markdown", "csv", "web".
            If None, inferred from file extension or URL prefix.
        **kwargs: Additional arguments passed to the specific loader.

    Returns:
        List of Document objects.
    """
    # Explicit source type
    if source_type:
        loaders = {
            "pdf": load_pdf,
            "markdown": load_markdown,
            "csv": load_csv,
            "web": load_web,
        }
        if source_type not in loaders:
            raise ValueError(f"Unknown source_type '{source_type}'. Choose from: {list(loaders)}")
        return loaders[source_type](source, **kwargs)

    # Auto-detect from URL
    if source.startswith(("http://", "https://")):
        return load_web(source, **kwargs)

    # Auto-detect from file extension
    path = Path(source)
    ext = path.suffix.lower()
    if ext in EXTENSION_MAP:
        return EXTENSION_MAP[ext](source, **kwargs)

    raise ValueError(
        f"Cannot infer source type from '{source}'. "
        f"Supported extensions: {list(EXTENSION_MAP)}. "
        "For web pages, use a URL starting with http:// or https://. "
        "Or specify source_type explicitly."
    )
