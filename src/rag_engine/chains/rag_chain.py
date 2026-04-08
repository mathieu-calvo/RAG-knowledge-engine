from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from rag_engine.chains.prompts import RAG_PROMPT


def format_documents(docs) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate | None = None,
) -> Runnable:
    """Build a RAG chain using LangChain Expression Language (LCEL).

    The chain: question -> retriever -> format docs -> prompt -> LLM -> string output

    Args:
        retriever: A LangChain retriever for fetching relevant documents.
        llm: A LangChain chat model.
        prompt: Prompt template. Defaults to RAG_PROMPT.

    Returns:
        A Runnable chain that takes {"question": str} and returns a string answer.
    """
    prompt = prompt or RAG_PROMPT

    chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def build_rag_chain_with_sources(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate | None = None,
) -> Runnable:
    """Build a RAG chain that returns both the answer and source documents.

    Args:
        retriever: A LangChain retriever.
        llm: A LangChain chat model.
        prompt: Prompt template. Defaults to RAG_PROMPT.

    Returns:
        A Runnable chain that takes {"question": str} and returns
        {"answer": str, "source_documents": list[Document]}.
    """
    prompt = prompt or RAG_PROMPT

    def retrieve_and_format(question: str):
        docs = retriever.invoke(question)
        return {"documents": docs, "context": format_documents(docs)}

    def run_chain(inputs):
        retrieval = retrieve_and_format(inputs["question"])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {"context": retrieval["context"], "question": inputs["question"]}
        )
        return {
            "answer": answer,
            "source_documents": retrieval["documents"],
        }

    return RunnablePassthrough() | run_chain
