from __future__ import annotations

import pandas as pd
import pytest

from seshat.core.models.enums import ConceptType
from seshat.eval.models import RetrievalCorpusExample, RetrievalCorpusNode
from seshat.eval.ranked_search_runner import _aggregate_metrics, _build_dataframe
from tests.unit.eval.helpers import make_eval_result


def _make_retrieval_node(
    node_id: str,
    concept_type: ConceptType = ConceptType.DECISION,
    title: str = "Title",
    description: str = "Description",
) -> RetrievalCorpusNode:
    return RetrievalCorpusNode(
        id=node_id,
        type=concept_type,
        title=title,
        description=description,
        quote="some quote",
    )


def _make_retrieval_example(
    corpus_id: str = "ret-1",
    query_type: ConceptType = ConceptType.DECISION,
    candidate_types: list[ConceptType] | None = None,
) -> RetrievalCorpusExample:
    if candidate_types is None:
        candidate_types = [ConceptType.RISK]
    query = _make_retrieval_node("query-node", query_type, title="Query title", description="Query desc")
    candidates = [
        _make_retrieval_node(f"cand-{i}", t, title=f"Cand {i}", description=f"Cand desc {i}")
        for i, t in enumerate(candidate_types)
    ]
    return RetrievalCorpusExample(
        corpus_id=corpus_id,
        description="test retrieval example",
        query_node=query,
        candidate_nodes=candidates,
        expected_relevant_ids=[f"cand-{i}" for i in range(len(candidate_types))],
    )


class TestBuildDataframe:
    def test_returns_dataframe_with_one_row_per_example(self):
        examples = [_make_retrieval_example("r1"), _make_retrieval_example("r2")]
        df = _build_dataframe(examples)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_inputs_contains_corpus_id_and_slimmed_nodes(self):
        ex = _make_retrieval_example("my-ret")
        df = _build_dataframe([ex])
        inputs = df.iloc[0]["inputs"]

        assert inputs["corpus_id"] == "my-ret"
        assert inputs["_query_node"] == {
            "id": "query-node",
            "type": "decision",
            "title": "Query title",
            "description": "Query desc",
        }
        assert inputs["_candidate_nodes"] == [
            {"id": "cand-0", "type": "risk", "title": "Cand 0", "description": "Cand desc 0"}
        ]

    def test_expectations_contain_expected_relevant_ids(self):
        ex = _make_retrieval_example("ret-x", candidate_types=[ConceptType.RISK, ConceptType.DECISION])
        df = _build_dataframe([ex])
        expectations = df.iloc[0]["expectations"]

        assert expectations["expected_relevant_ids"] == ["cand-0", "cand-1"]

    def test_tags_are_prefixed_with_corpus_dot(self):
        ex = RetrievalCorpusExample(
            corpus_id="tagged",
            description="d",
            query_node=_make_retrieval_node("q"),
            candidate_nodes=[_make_retrieval_node("c")],
            expected_relevant_ids=["c"],
            tags={"tier": "hard", "source": "manual"},
        )
        df = _build_dataframe([ex])
        tags = df.iloc[0]["tags"]

        assert tags == {"corpus.tier": "hard", "corpus.source": "manual"}

    def test_empty_examples_produces_empty_dataframe(self):
        df = _build_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestAggregateMetrics:
    def test_extracts_recall_at_5_and_precision_at_5(self):
        eval_result = make_eval_result(
            {
                "recall_at_5/mean": 0.8,
                "precision_at_5/mean": 0.6,
            }
        )
        result = _aggregate_metrics(eval_result)

        assert result == {"recall_at_5": 0.8, "precision_at_5": 0.6}

    def test_extracts_mrr_at_5(self):
        eval_result = make_eval_result(
            {
                "recall_at_5/mean": 0.8,
                "precision_at_5/mean": 0.6,
                "mrr_at_5/mean": 0.9,
            }
        )
        result = _aggregate_metrics(eval_result)

        assert result["mrr_at_5"] == pytest.approx(0.9)

    def test_absent_mrr_at_5_is_excluded(self):
        eval_result = make_eval_result({"recall_at_5/mean": 1.0, "precision_at_5/mean": 0.5})
        result = _aggregate_metrics(eval_result)

        assert "mrr_at_5" not in result

    def test_absent_metrics_are_excluded(self):
        eval_result = make_eval_result(
            {
                "recall_at_5/mean": 1.0,
                # precision_at_5 absent
            }
        )
        result = _aggregate_metrics(eval_result)

        assert "recall_at_5" in result
        assert "precision_at_5" not in result

    def test_returns_floats(self):
        eval_result = make_eval_result(
            {
                "recall_at_5/mean": 1,
                "precision_at_5/mean": 0,
                "mrr_at_5/mean": 1,
            }
        )
        result = _aggregate_metrics(eval_result)

        assert isinstance(result["recall_at_5"], float)
        assert isinstance(result["precision_at_5"], float)
        assert isinstance(result["mrr_at_5"], float)

    def test_empty_metrics_returns_empty_dict(self):
        eval_result = make_eval_result({})
        assert _aggregate_metrics(eval_result) == {}
