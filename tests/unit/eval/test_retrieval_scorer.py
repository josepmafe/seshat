import pytest

from seshat.eval.scorers import retrieval_scorer


class TestRetrievalScorer:
    def test_perfect_recall(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-1", "id-2"]},
            expectations={"expected_relevant_ids": ["id-1", "id-2"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(1.0)

    def test_missed_relevant_node(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-2"]},
            expectations={"expected_relevant_ids": ["id-1", "id-2"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(0.5)

    def test_no_relevant_nodes_retrieved(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-3"]},
            expectations={"expected_relevant_ids": ["id-1"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(0.0)

    def test_precision_at_5(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-1", "id-2", "id-3"]},
            expectations={"expected_relevant_ids": ["id-1"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["precision_at_5"] == pytest.approx(1 / 3)

    def test_empty_expected_returns_perfect(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": []},
            expectations={"expected_relevant_ids": []},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(1.0)
        assert by_name["precision_at_5"] == pytest.approx(1.0)
