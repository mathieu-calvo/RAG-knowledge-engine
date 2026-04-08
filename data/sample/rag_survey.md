# Retrieval-Augmented Generation: A Survey

## Introduction

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge at inference time. Instead of relying solely on the knowledge encoded in model weights during training, RAG systems retrieve pertinent documents from a knowledge base and include them as context in the prompt.

This approach addresses several fundamental limitations of LLMs:

- **Knowledge cutoff**: LLMs only know what was in their training data. RAG allows them to access up-to-date information.
- **Hallucination**: Without grounding, LLMs may generate plausible-sounding but incorrect information. RAG provides factual grounding.
- **Domain specificity**: RAG enables LLMs to answer questions about proprietary or specialized data they were never trained on.

## Core Components

A typical RAG pipeline consists of the following stages:

### 1. Document Ingestion

Raw documents (PDFs, web pages, databases) are loaded and converted into a standardized text format. Each document is represented as a text string paired with metadata (source, page number, date, etc.).

### 2. Chunking

Documents are split into smaller pieces called "chunks." The chunk size is a critical hyperparameter:
- **Too large**: Retrieved chunks may contain irrelevant information, diluting the useful context.
- **Too small**: Chunks may lack sufficient context to be meaningful on their own.

Common strategies include fixed-size splitting, recursive splitting (respecting paragraph and sentence boundaries), and semantic splitting (grouping sentences by embedding similarity).

### 3. Embedding

Each chunk is converted into a dense vector (embedding) using an embedding model. These vectors capture the semantic meaning of the text, enabling similarity-based retrieval. Popular embedding models include OpenAI's text-embedding-3-small and open-source models like BGE and E5.

### 4. Vector Storage

Embeddings are stored in a vector database (e.g., ChromaDB, Pinecone, FAISS) that supports efficient nearest-neighbor search. The database indexes the vectors using algorithms like HNSW (Hierarchical Navigable Small World) for fast retrieval.

### 5. Retrieval

When a user asks a question, the query is embedded using the same model, and the most similar chunks are retrieved from the vector store. Retrieval strategies include:
- **Similarity search**: Return the top-k most similar chunks by cosine distance.
- **MMR (Maximal Marginal Relevance)**: Balance relevance with diversity to avoid returning near-duplicate chunks.
- **Hybrid search**: Combine dense vector search with sparse keyword search (BM25) for broader coverage.

### 6. Generation

The retrieved chunks are inserted into a prompt template alongside the user's question and sent to the LLM. The model generates an answer grounded in the provided context.

## Advanced Techniques

### Query Transformation

Instead of using the raw user query for retrieval, transform it to improve results:
- **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer first, embed it, and use that embedding for retrieval.
- **Multi-query**: Generate multiple query variations and retrieve for each, then merge results.

### Re-ranking

After initial retrieval of a larger candidate set (e.g., top-20), use a cross-encoder model to re-score and re-rank the chunks, keeping only the most relevant (e.g., top-5). Cross-encoders are more accurate than bi-encoders but too slow for the initial retrieval pass.

### Evaluation

RAG systems should be evaluated on multiple dimensions:
- **Faithfulness**: Does the answer only contain information supported by the retrieved context?
- **Answer relevancy**: Is the answer actually relevant to the question asked?
- **Context precision**: Are the retrieved documents relevant to the question?
- **Context recall**: Are all the relevant documents in the knowledge base actually retrieved?

The RAGAS framework provides automated metrics for all of these dimensions.

## References

- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."
- Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey."
