from langchain_core.language_models import BaseChatModel

from rag_engine.config import get_settings


class LLMProvider:
    """Factory for creating swappable LLM instances.

    Supports OpenAI and Anthropic models through a unified interface.
    All returned models implement LangChain's BaseChatModel, so they
    can be used interchangeably in chains and prompts.

    Usage:
        llm = LLMProvider.get_llm()                              # Uses config defaults
        llm = LLMProvider.get_llm(provider="anthropic")          # Override provider
        llm = LLMProvider.get_llm(provider="openai", model="gpt-4o")  # Override both
    """

    SUPPORTED_PROVIDERS = ("openai", "anthropic")

    @staticmethod
    def get_llm(
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> BaseChatModel:
        """Create an LLM instance for the specified provider.

        Args:
            provider: LLM provider ("openai" or "anthropic"). Defaults to config.
            model: Model name. Defaults to config.
            temperature: Sampling temperature. Defaults to 0.0 for deterministic output.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            A LangChain BaseChatModel instance.

        Raises:
            ValueError: If the provider is not supported.
        """
        settings = get_settings()
        provider = provider or settings.llm_provider
        model = model or settings.llm_model

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=settings.openai_api_key,
                temperature=temperature,
                **kwargs,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model,
                api_key=settings.anthropic_api_key,
                temperature=temperature,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: {LLMProvider.SUPPORTED_PROVIDERS}"
            )
