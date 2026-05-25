import pytest
from mlflow.entities import Feedback

from seshat.eval.models import IdentificationCorpusNode
from seshat.eval.scorers import identification_scorer
from seshat.models.enums import ConceptType
from tests.helpers import make_node

TRANSCRIPT = (
    "We decided to use PostgreSQL for all operational data. There is a risk that replication lag could affect reads."
)


def _corpus_node(ctype: ConceptType, quote: str) -> IdentificationCorpusNode:
    return IdentificationCorpusNode(quote=quote, type=ctype, title="T", description="A description.")


class TestIdentificationScorer:
    def test_returns_feedback_list(self):
        quote = "We decided to use PostgreSQL for all operational data."
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.DECISION).model_dump(mode="json")]
        }
        expectations = {"expected_nodes": [_corpus_node(ConceptType.DECISION, quote).model_dump(mode="json")]}
        feedbacks = identification_scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        assert isinstance(feedbacks, list)
        assert all(isinstance(f, Feedback) for f in feedbacks)

    def test_perfect_precision_recall_decision(self):
        quote = "We decided to use PostgreSQL for all operational data."
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.DECISION).model_dump(mode="json")]
        }
        expectations = {"expected_nodes": [_corpus_node(ConceptType.DECISION, quote).model_dump(mode="json")]}
        feedbacks = identification_scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["decision.precision"] == pytest.approx(1.0)
        assert by_name["decision.recall"] == pytest.approx(1.0)

    def test_missed_node_gives_zero_recall(self):
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {"nodes": []}
        expectations = {
            "expected_nodes": [
                _corpus_node(ConceptType.RISK, "There is a risk that replication lag could affect reads.").model_dump(
                    mode="json"
                )
            ]
        }
        feedbacks = identification_scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["risk.recall"] == pytest.approx(0.0)

    def test_spurious_node_gives_zero_precision(self):
        quote = "We decided to use PostgreSQL for all operational data."
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.ACTION_ITEM).model_dump(mode="json")
            ]
        }
        expectations = {"expected_nodes": []}
        feedbacks = identification_scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(0.0)
