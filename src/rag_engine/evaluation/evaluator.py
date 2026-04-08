from dataclasses import dataclass

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from rag_engine.chains.rag_chain import build_rag_chain


@dataclass
class EvalSample:
    """A single evaluation sample with question, expected answer, and optional contexts."""

    question: str
    ground_truth: str
    contexts: list[str] | None = None
    answer: str | None = None


def run_rag_on_eval_set(
    eval_samples: list[EvalSample],
    retriever: BaseRetriever,
    llm: BaseChatModel,
) -> list[EvalSample]:
    """Run the RAG chain on each evaluation sample, populating the answer and contexts fields.

    Args:
        eval_samples: List of EvalSample with question and ground_truth filled in.
        retriever: The retriever to use.
        llm: The LLM to use.

    Returns:
        The same list with answer and contexts populated.
    """
    chain = build_rag_chain(retriever, llm)

    for sample in eval_samples:
        # Get answer from RAG chain
        sample.answer = chain.invoke(sample.question)

        # Get retrieved contexts separately
        docs = retriever.invoke(sample.question)
        sample.contexts = [doc.page_content for doc in docs]

    return eval_samples


def evaluate_with_ragas(eval_samples: list[EvalSample]) -> pd.DataFrame:
    """Evaluate RAG results using the RAGAS framework.

    Requires the 'ragas' package: pip install ragas

    Args:
        eval_samples: List of EvalSample with all fields populated.

    Returns:
        DataFrame with per-sample and aggregate RAGAS scores.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Convert to RAGAS dataset format
    data = {
        "question": [s.question for s in eval_samples],
        "answer": [s.answer for s in eval_samples],
        "contexts": [s.contexts for s in eval_samples],
        "ground_truth": [s.ground_truth for s in eval_samples],
    }

    dataset = Dataset.from_dict(data)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return results.to_pandas()


def simple_evaluate(eval_samples: list[EvalSample]) -> pd.DataFrame:
    """Simple evaluation without RAGAS (no extra dependencies needed).

    Computes basic metrics:
    - has_answer: Whether the RAG system produced a non-empty answer.
    - answer_length: Length of the generated answer.
    - num_contexts: Number of context chunks retrieved.

    Args:
        eval_samples: List of EvalSample with answer and contexts populated.

    Returns:
        DataFrame with per-sample metrics.
    """
    rows = []
    for sample in eval_samples:
        rows.append(
            {
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "answer": sample.answer or "",
                "has_answer": bool(sample.answer and sample.answer.strip()),
                "answer_length": len(sample.answer or ""),
                "num_contexts": len(sample.contexts or []),
            }
        )

    return pd.DataFrame(rows)
