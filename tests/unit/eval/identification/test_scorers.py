from datetime import date

import pytest

from seshat.eval.identification.scorers import scorer
from seshat.models.enums import ConceptType, IngestionSource, NodeStatus
from seshat.models.nodes import NodeMetadata
from tests.helpers import make_node
from tests.unit.eval.identification.helpers import corpus_node

TRANSCRIPT = (
    "We decided to use PostgreSQL for all operational data. There is a risk that replication lag could affect reads."
)

_DECISION_QUOTE = "We decided to use PostgreSQL for all operational data."
_RISK_QUOTE = "There is a risk that replication lag could affect reads."


def _node_with_concept_fields(quote: str, ctype: ConceptType, concept_fields: dict) -> dict:
    metadata = NodeMetadata(
        job_id="job-1",
        meeting_date=date(2026, 4, 21),
        ingestion_source=IngestionSource.PIPELINE,
        concept_fields=concept_fields,
    )
    return make_node(
        quote=quote,
        transcript=TRANSCRIPT,
        type=ctype,
        status=NodeStatus.APPROVED,
        metadata=metadata,
    ).model_dump(mode="json")


class TestIdentificationScorer:
    def test_perfect_precision_recall_decision(self):
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=_DECISION_QUOTE, transcript=TRANSCRIPT, type=ConceptType.DECISION).model_dump(
                    mode="json"
                )
            ]
        }
        expectations = {"expected_nodes": [corpus_node(_DECISION_QUOTE, ConceptType.DECISION).model_dump(mode="json")]}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["decision.precision"] == pytest.approx(1.0)
        assert by_name["decision.recall"] == pytest.approx(1.0)

    def test_missed_node_gives_zero_recall(self):
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {"nodes": []}
        expectations = {"expected_nodes": [corpus_node(_RISK_QUOTE, ConceptType.RISK).model_dump(mode="json")]}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["risk.recall"] == pytest.approx(0.0)

    def test_spurious_node_gives_zero_precision(self):
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=_DECISION_QUOTE, transcript=TRANSCRIPT, type=ConceptType.ACTION_ITEM).model_dump(
                    mode="json"
                )
            ]
        }
        expectations = {"expected_nodes": []}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(0.0)

    def test_mixed_type_decision_hit_and_risk_miss(self):
        # DECISION matches → precision=1, recall=1; RISK expected but not predicted → recall=0
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=_DECISION_QUOTE, transcript=TRANSCRIPT, type=ConceptType.DECISION).model_dump(
                    mode="json"
                )
            ]
        }
        expectations = {
            "expected_nodes": [
                corpus_node(_DECISION_QUOTE, ConceptType.DECISION).model_dump(mode="json"),
                corpus_node(_RISK_QUOTE, ConceptType.RISK).model_dump(mode="json"),
            ]
        }
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["decision.precision"] == pytest.approx(1.0)
        assert by_name["decision.recall"] == pytest.approx(1.0)
        assert by_name["risk.recall"] == pytest.approx(0.0)
        assert "action_item.precision" not in by_name
        assert "open_question.precision" not in by_name


class TestFieldAccuracyFeedback:
    @pytest.mark.parametrize(
        ("quote", "ctype", "predicted_fields", "expected_fields", "expected_key", "expected_score"),
        [
            (
                _DECISION_QUOTE,
                ConceptType.DECISION,
                {"alternatives_considered": ["MySQL", "SQLite"]},
                {"alternatives_considered": ["MySQL", "SQLite"]},
                "decision.alternatives_considered",
                1.0,
            ),
            (
                _DECISION_QUOTE,
                ConceptType.DECISION,
                {"alternatives_considered": ["MySQL"]},
                {"alternatives_considered": ["MySQL", "SQLite"]},
                "decision.alternatives_considered",
                0.5,
            ),
            (_RISK_QUOTE, ConceptType.RISK, {"type": "future"}, {"type": "future"}, "risk.type", 1.0),
            (_RISK_QUOTE, ConceptType.RISK, {"type": "blocker"}, {"type": "future"}, "risk.type", 0.0),
            (
                _DECISION_QUOTE,
                ConceptType.ACTION_ITEM,
                {"assignee": "Alice"},
                {"assignee": "Alice"},
                "action_item.assignee",
                1.0,
            ),
        ],
    )
    def test_field_accuracy(self, quote, ctype, predicted_fields, expected_fields, expected_key, expected_score):
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {"nodes": [_node_with_concept_fields(quote, ctype, predicted_fields)]}
        expectations = {
            "expected_nodes": [corpus_node(quote, ctype, extra_fields=expected_fields).model_dump(mode="json")]
        }
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name[expected_key] == pytest.approx(expected_score)


class TestNegativeCheckFeedback:
    def test_spurious_rate_one_when_prediction_on_negative_example(self):
        # DECISION predicted but nothing expected → spurious_rate = 1.0
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=_DECISION_QUOTE, transcript=TRANSCRIPT, type=ConceptType.DECISION).model_dump(
                    mode="json"
                )
            ]
        }
        expectations = {"expected_nodes": []}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["decision.spurious_rate"] == pytest.approx(1.0)

    def test_spurious_rate_zero_when_no_prediction_on_negative_example(self):
        # No nodes predicted, nothing expected → spurious_rate = 0.0 for all types
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {"nodes": []}
        expectations = {"expected_nodes": []}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        for ctype in ConceptType:
            assert by_name[f"{ctype.value}.spurious_rate"] == pytest.approx(0.0)

    def test_spurious_rate_only_for_negative_types_when_mixed(self):
        # DECISION has expected node, RISK does not — only RISK gets spurious_rate
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=_DECISION_QUOTE, transcript=TRANSCRIPT, type=ConceptType.DECISION).model_dump(
                    mode="json"
                ),
                make_node(quote=_RISK_QUOTE, transcript=TRANSCRIPT, type=ConceptType.RISK).model_dump(mode="json"),
            ]
        }
        expectations = {"expected_nodes": [corpus_node(_DECISION_QUOTE, ConceptType.DECISION).model_dump(mode="json")]}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert "decision.spurious_rate" not in by_name
        assert by_name["risk.spurious_rate"] == pytest.approx(1.0)
        assert by_name["action_item.spurious_rate"] == pytest.approx(0.0)
        assert by_name["open_question.spurious_rate"] == pytest.approx(0.0)
