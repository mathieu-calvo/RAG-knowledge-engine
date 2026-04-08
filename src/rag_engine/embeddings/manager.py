from langchain_openai import OpenAIEmbeddings

from rag_engine.config import get_settings


def get_embedding_model(model: str | None = None) -> OpenAIEmbeddings:
    """Create an embedding model instance.

    Currently supports OpenAI embedding models. The embedding model is separate
    from the LLM provider choice -- embeddings always use OpenAI since they offer
    the best price/performance ratio for embeddings.

    Args:
        model: Embedding model name. Defaults to config value.

    Returns:
        An OpenAIEmbeddings instance.
    """
    settings = get_settings()
    model = model or settings.embedding_model

    return OpenAIEmbeddings(
        model=model,
        api_key=settings.openai_api_key,
    )
