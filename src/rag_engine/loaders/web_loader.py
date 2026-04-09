import time

import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_web(url: str, timeout: int = 30, retries: int = 2) -> list[Document]:
    """Load a web page and return a list of Documents.

    Args:
        url: The URL of the web page to load.
        timeout: Per-request timeout in seconds.
        retries: Number of additional attempts after the first failure.

    Returns:
        List of Document objects with the page text and metadata.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")

    loader = WebBaseLoader(
        url,
        requests_kwargs={"timeout": timeout},
        header_template={
            "User-Agent": (
                "RAG-knowledge-engine/0.1 "
                "(https://github.com/; educational project)"
            )
        },
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            documents = loader.load()
            break
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            if attempt < retries:
                time.sleep(2**attempt)
    else:
        raise ConnectionError(
            f"Failed to fetch {url} after {retries + 1} attempts "
            f"(timeout={timeout}s). Check your network/VPN/proxy. "
            f"Last error: {last_error}"
        ) from last_error

    for doc in documents:
        doc.metadata["source"] = url
        doc.metadata["file_type"] = "web"

    return documents
