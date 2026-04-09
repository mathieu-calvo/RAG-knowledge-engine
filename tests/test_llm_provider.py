import pytest

from rag_engine.config import get_settings
from rag_engine.llm.provider import LLMProvider


@pytest.fixture
def fake_google_key(monkeypatch):
    """Provide a fake GOOGLE_API_KEY so ChatGoogleGenerativeAI passes its
    construction-time validation. No network calls are made in these tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestLLMProvider:
    def test_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMProvider.get_llm(provider="unsupported")

    def test_google_provider_returns_chat_model(self, fake_google_key):
        """Verify that requesting Google returns the correct class."""
        llm = LLMProvider.get_llm(provider="google", model="gemini-2.5-flash")
        from langchain_google_genai import ChatGoogleGenerativeAI

        assert isinstance(llm, ChatGoogleGenerativeAI)

    @pytest.mark.requires_api_key
    def test_openai_provider_returns_chat_model(self):
        """Verify that requesting OpenAI returns the correct class."""
        llm = LLMProvider.get_llm(provider="openai", model="gpt-4o-mini")
        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)

    @pytest.mark.requires_api_key
    def test_anthropic_provider_returns_chat_model(self):
        """Verify that requesting Anthropic returns the correct class."""
        llm = LLMProvider.get_llm(
            provider="anthropic", model="claude-sonnet-4-20250514"
        )
        from langchain_anthropic import ChatAnthropic

        assert isinstance(llm, ChatAnthropic)

    def test_temperature_is_set(self, fake_google_key):
        llm = LLMProvider.get_llm(
            provider="google", model="gemini-2.5-flash", temperature=0.7
        )
        assert llm.temperature == 0.7
