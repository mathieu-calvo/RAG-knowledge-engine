from pathlib import Path

from langchain_core.documents import Document


def load_markdown(file_path: str | Path) -> list[Document]:
    """Load a Markdown file and return a list of Documents.

    Uses a simple file read (no external dependencies like 'unstructured').

    Args:
        file_path: Path to the Markdown file.

    Returns:
        List of Document objects with the markdown content and metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    if file_path.suffix.lower() not in (".md", ".markdown"):
        raise ValueError(f"Expected a .md/.markdown file, got: {file_path.suffix}")

    text = file_path.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={"source": str(file_path), "file_type": "markdown"},
    )

    return [document]
