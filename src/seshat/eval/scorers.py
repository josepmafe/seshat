from __future__ import annotations

from collections import defaultdict

import mlflow.genai
from mlflow.entities import Feedback

from seshat.eval.matcher import MatchedNode, MatchResult, match_nodes
from seshat.eval.models import IdentificationCorpusNode
from seshat.models.enums import ConceptType
from seshat.models.nodes import KBNode


@mlflow.genai.scorer
def identification_scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """Deterministic precision/recall scorer for identification quality. No LLM calls."""
    transcript = inputs["transcript"]
    expected = [IdentificationCorpusNode(**n) for n in expectations["expected_nodes"]]
    predicted = [KBNode(**n) for n in outputs["nodes"]]

    result = match_nodes(transcript, expected, predicted)
    feedbacks = _precision_recall_feedback(result)
    feedbacks += _field_accuracy_feedback(result.matched)
    return feedbacks


def _precision_recall_feedback(result: MatchResult) -> list[Feedback]:
    true_positives: dict[ConceptType, int] = defaultdict(int)
    false_positives: dict[ConceptType, int] = defaultdict(int)
    false_negatives: dict[ConceptType, int] = defaultdict(int)

    for match in result.matched:
        true_positives[match.expected.type] += 1

    for node in result.spurious:
        false_positives[node.type] += 1

    for node in result.missed:
        false_negatives[node.type] += 1

    feedbacks: list[Feedback] = []
    for ctype in ConceptType:
        predicted_count = true_positives[ctype] + false_positives[ctype]
        gold_count = true_positives[ctype] + false_negatives[ctype]
        if predicted_count == 0 and gold_count == 0:
            continue

        precision = true_positives[ctype] / predicted_count if predicted_count else 0.0
        recall = true_positives[ctype] / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        feedbacks.extend(
            [
                Feedback(name=f"{ctype}.precision", value=precision),
                Feedback(name=f"{ctype}.recall", value=recall),
                Feedback(name=f"{ctype}.f1", value=f1),
            ]
        )
    return feedbacks


def _field_accuracy_feedback(matched: list[MatchedNode]) -> list[Feedback]:
    # TODO: implement per-type field accuracy (assignee, due, rationale, risk.type, context)
    # using rapidfuzz token_set_ratio against IdentificationCorpusNode.extra_fields
    return []


def _nli_faithfulness_feedback(result: MatchResult) -> list[Feedback]:
    # TODO: implement using a local cross-encoder NLI model
    return []


@mlflow.genai.scorer
def resolution_scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """Precision/recall scorer for resolution quality. TP = (source_id, target_id, rel_type) exact match."""
    slug_to_uuid: dict[str, str] = expectations["slug_to_uuid"]
    expected_triples: set[tuple[str, str, str]] = {
        (slug_to_uuid[r["source"]], slug_to_uuid[r["target"]], r["rel_type"])
        for r in expectations["expected_relations"]
        if r["source"] in slug_to_uuid and r["target"] in slug_to_uuid
    }
    predicted_triples: set[tuple[str, str, str]] = {
        (str(r["source_id"]), str(r["target_id"]), r["rel_type"]) for r in outputs["relationships"]
    }

    tp = len(expected_triples & predicted_triples)
    fp = len(predicted_triples - expected_triples)
    fn = len(expected_triples - predicted_triples)

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not expected_triples else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return [
        Feedback(name="precision", value=precision),
        Feedback(name="recall", value=recall),
        Feedback(name="f1", value=f1),
    ]


@mlflow.genai.scorer
def retrieval_scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """recall@5 and precision@5 for retrieval quality."""
    retrieved_ids: list[str] = outputs.get("retrieved_ids", [])
    expected_ids: set[str] = set(expectations.get("expected_relevant_ids", []))

    if not expected_ids:
        return [Feedback(name="recall_at_5", value=1.0), Feedback(name="precision_at_5", value=1.0)]

    top_5 = set(retrieved_ids[:5])
    recall = len(expected_ids & top_5) / len(expected_ids)
    precision = len(expected_ids & top_5) / len(top_5) if top_5 else 0.0

    return [
        Feedback(name="recall_at_5", value=recall),
        Feedback(name="precision_at_5", value=precision),
    ]
