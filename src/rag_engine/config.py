from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the RAG Knowledge Engine.

    All settings can be overridden via environment variables or a .env file.
    """

    # LLM API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # LLM Configuration
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Vector Store
    chroma_persist_dir: str = "./chroma_db"

    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    retrieval_strategy: str = "similarity"  # "similarity", "mmr"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
