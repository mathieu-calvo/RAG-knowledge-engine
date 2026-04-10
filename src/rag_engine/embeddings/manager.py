from langchain_core.embeddings import Embeddings

from rag_engine.config import get_settings


def get_embedding_model(
    provider: str | None = None,
    model: str | None = None,
) -> Embeddings:
    """Create an embedding model instance.

    Supports two providers:
    - "huggingface" (default): Runs locally via sentence-transformers. Free, no API key.
    - "openai": Uses OpenAI's API. Requires an API key and costs money.

    Args:
        provider: Embedding provider ("huggingface" or "openai"). Defaults to config.
        model: Model name. Defaults to config value.

    Returns:
        A LangChain Embeddings instance.
    """
    settings = get_settings()
    provider = provider or settings.embedding_provider
    model = model or settings.embedding_model

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model)

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openai_api_key,
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider '{provider}'. "
            "Choose from: 'huggingface', 'openai'"
        )
