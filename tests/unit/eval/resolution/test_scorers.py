import pytest

from seshat.core.models.enums import ConceptType
from seshat.eval.resolution.scorers import scorer

# src node is action_item; tgt is decision — slug-based, matching how the runner serialises
_SLUG_TO_TYPE = {"src": ConceptType.ACTION_ITEM, "tgt": ConceptType.DECISION}


def _rel(source: str, target: str, rel_type: str) -> dict:
    return {"source": source, "target": target, "rel_type": rel_type}


def _expectations(relations: list[dict]) -> dict:
    return {
        "expected_relations": relations,
        "slug_to_type": _SLUG_TO_TYPE,
    }


class TestResolutionScorer:
    def test_both_empty_no_feedbacks(self):
        # tp=fp=fn=0 for all types: no data to score
        feedbacks = scorer(
            inputs={},
            outputs={"relations": []},
            expectations=_expectations([]),
        )
        assert feedbacks == []

    def test_perfect_precision_recall(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relations": [_rel("src", "tgt", "amends")]},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name: dict[str, float] = {f.name: float(f.value) for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(1.0)
        assert by_name["action_item.recall"] == pytest.approx(1.0)

    def test_missed_relation_zero_recall(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relations": []},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name: dict[str, float] = {f.name: float(f.value) for f in feedbacks}
        assert by_name["action_item.recall"] == pytest.approx(0.0)

    def test_spurious_relation_zero_precision(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relations": [_rel("src", "tgt", "supersedes")]},
            expectations=_expectations([]),
        )
        by_name: dict[str, float] = {f.name: float(f.value) for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(0.0)

    def test_wrong_rel_type_is_fp_and_fn(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relations": [_rel("src", "tgt", "supersedes")]},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name: dict[str, float] = {f.name: float(f.value) for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(0.0)
        assert by_name["action_item.recall"] == pytest.approx(0.0)
