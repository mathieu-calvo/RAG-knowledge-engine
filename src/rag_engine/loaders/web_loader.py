from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_web(url: str) -> list[Document]:
    """Load a web page and return a list of Documents.

    Args:
        url: The URL of the web page to load.

    Returns:
        List of Document objects with the page text and metadata.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")

    loader = WebBaseLoader(url)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = url
        doc.metadata["file_type"] = "web"

    return documents
