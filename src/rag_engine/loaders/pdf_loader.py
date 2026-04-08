from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF file and return a list of Documents (one per page).

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects with page content and metadata
        including source path and page number.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    # Ensure consistent metadata
    for doc in documents:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_type"] = "pdf"

    return documents
