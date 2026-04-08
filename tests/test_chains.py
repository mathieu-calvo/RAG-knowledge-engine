
from langchain_core.documents import Document

from rag_engine.chains.prompts import HYDE_PROMPT, RAG_PROMPT, RAG_PROMPT_CONCISE
from rag_engine.chains.rag_chain import format_documents


class TestFormatDocuments:
    def test_format_single_document(self):
        docs = [Document(page_content="Hello world.", metadata={"source": "test.pdf"})]
        result = format_documents(docs)
        assert "[Source 1: test.pdf]" in result
        assert "Hello world." in result

    def test_format_multiple_documents(self):
        docs = [
            Document(page_content="First.", metadata={"source": "a.pdf"}),
            Document(page_content="Second.", metadata={"source": "b.md"}),
        ]
        result = format_documents(docs)
        assert "[Source 1: a.pdf]" in result
        assert "[Source 2: b.md]" in result
        assert "First." in result
        assert "Second." in result

    def test_format_missing_source(self):
        docs = [Document(page_content="No source.", metadata={})]
        result = format_documents(docs)
        assert "[Source 1: Unknown]" in result


class TestPrompts:
    def test_rag_prompt_has_required_variables(self):
        assert "context" in RAG_PROMPT.input_variables
        assert "question" in RAG_PROMPT.input_variables

    def test_concise_prompt_has_required_variables(self):
        assert "context" in RAG_PROMPT_CONCISE.input_variables
        assert "question" in RAG_PROMPT_CONCISE.input_variables

    def test_hyde_prompt_has_question(self):
        assert "question" in HYDE_PROMPT.input_variables
