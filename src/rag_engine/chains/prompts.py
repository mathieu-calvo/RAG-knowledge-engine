from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Standard RAG prompt: answer based on context, cite sources, admit uncertainty
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use ONLY the information from the context below to answer the question. "
            "If the context doesn't contain enough information to answer, say so clearly. "
            "When possible, reference which source document your answer comes from.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# Concise variant for shorter answers
RAG_PROMPT_CONCISE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the provided context. Be concise (1-3 sentences). "
            "If the context is insufficient, say 'I don't have enough information.'\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# Conversational RAG prompt with chat history
CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use ONLY the information from the context to answer. "
            "If the context is insufficient, say so. Reference source documents when possible.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# HyDE prompt: generate a hypothetical answer for better retrieval
HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the question below, write a short paragraph that would be a good answer "
            "to this question. This will be used to find relevant documents, so make it "
            "detailed and specific. Do not say 'I don't know'.",
        ),
        ("human", "{question}"),
    ]
)
