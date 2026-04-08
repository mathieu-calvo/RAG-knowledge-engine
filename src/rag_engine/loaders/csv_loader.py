from pathlib import Path

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document


def load_csv(file_path: str | Path, content_columns: list[str] | None = None) -> list[Document]:
    """Load a CSV file and return a list of Documents (one per row).

    Args:
        file_path: Path to the CSV file.
        content_columns: Optional list of column names to include in page_content.
            If None, all columns are included.

    Returns:
        List of Document objects with row content and metadata
        including source path and row index.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    if not file_path.suffix.lower() == ".csv":
        raise ValueError(f"Expected a .csv file, got: {file_path.suffix}")

    loader_kwargs = {"source_column": None}
    if content_columns:
        loader_kwargs["content_columns"] = content_columns

    loader = CSVLoader(file_path=str(file_path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_type"] = "csv"

    return documents
