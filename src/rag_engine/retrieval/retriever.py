from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from rag_engine.config import get_settings


class RetrieverFactory:
    """Factory for creating configured retrievers from a vector store.

    Supports multiple retrieval strategies:
    - "similarity": Standard top-k nearest neighbor search.
    - "mmr": Maximal Marginal Relevance for diverse results.
    - "similarity_score_threshold": Filter by minimum similarity score.
    """

    SUPPORTED_STRATEGIES = ("similarity", "mmr", "similarity_score_threshold")

    @staticmethod
    def create_retriever(
        vectorstore: VectorStore,
        strategy: str | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> BaseRetriever:
        """Create a retriever with the specified strategy.

        Args:
            vectorstore: The vector store to retrieve from.
            strategy: Retrieval strategy. Defaults to config value.
            top_k: Number of documents to retrieve. Defaults to config value.
            **kwargs: Additional arguments for the retriever.
                For "mmr": fetch_k (int), lambda_mult (float 0-1).
                For "similarity_score_threshold": score_threshold (float 0-1).

        Returns:
            A LangChain BaseRetriever instance.
        """
        settings = get_settings()
        strategy = strategy or settings.retrieval_strategy
        top_k = top_k or settings.retrieval_top_k

        if strategy not in RetrieverFactory.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {RetrieverFactory.SUPPORTED_STRATEGIES}"
            )

        search_kwargs = {"k": top_k, **kwargs}

        if strategy == "similarity":
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs,
            )
        elif strategy == "mmr":
            search_kwargs.setdefault("fetch_k", top_k * 4)
            search_kwargs.setdefault("lambda_mult", 0.5)
            return vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs,
            )
        elif strategy == "similarity_score_threshold":
            search_kwargs.setdefault("score_threshold", 0.7)
            return vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs,
            )
