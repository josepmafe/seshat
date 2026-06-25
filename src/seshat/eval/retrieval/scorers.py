from __future__ import annotations

import mlflow.genai
from mlflow.entities import Feedback

TOP_K = 5


@mlflow.genai.scorer
def scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """recall@5, precision@5, and mrr@5 for retrieval quality."""
    retrieved_ids: list[str] = outputs.get("retrieved_ids", [])
    expected_ids: set[str] = set(expectations.get("expected_relevant_ids", []))
    top_k = set(retrieved_ids[:TOP_K])

    if not expected_ids:
        # negative case: any retrieved result is a false positive
        recall = 0.0 if top_k else 1.0
        return [Feedback(name="recall_at_5", value=recall)]

    recall = len(expected_ids & top_k) / len(expected_ids)
    precision = len(expected_ids & top_k) / TOP_K if top_k else 0.0
    mrr = next(
        (1.0 / (rank + 1) for rank, node_id in enumerate(retrieved_ids[:TOP_K]) if node_id in expected_ids),
        0.0,
    )

    return [
        Feedback(name="recall_at_5", value=recall),
        Feedback(name="precision_at_5", value=precision),
        Feedback(name="mrr_at_5", value=mrr),
    ]
