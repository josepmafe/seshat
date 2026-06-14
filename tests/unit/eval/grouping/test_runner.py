from __future__ import annotations

import pandas as pd

from seshat.eval.grouping.corpus_loader import GroupingCorpusExample, GroupingCorpusItem
from seshat.eval.grouping.runner import _aggregate_metrics, _build_anchored_concepts, _build_dataframe
from tests.unit.eval.helpers import make_eval_result


def _item(item_id: str, description: str = "Some description", title: str = "Title") -> GroupingCorpusItem:
    return GroupingCorpusItem(id=item_id, title=title, description=description)


def _example(corpus_id: str = "ex-1", n_items: int = 2) -> GroupingCorpusExample:
    items = [_item(f"item-{i}", description=f"Description {i}") for i in range(n_items)]
    expected_groups = [[item.id for item in items]]
    return GroupingCorpusExample(
        corpus_id=corpus_id,
        description="test example",
        items=items,
        expected_groups=expected_groups,
    )


class TestBuildAnchoredConcepts:
    def test_title_is_set_to_item_id(self):
        item = _item("kafka-choice", description="Chose Kafka as message broker")
        result = _build_anchored_concepts([item])

        assert len(result) == 1
        assert result[0].item.title == "kafka-choice"

    def test_description_is_preserved(self):
        item = _item("some-id", description="Detailed description text")
        result = _build_anchored_concepts([item])

        assert result[0].item.description == "Detailed description text"

    def test_quote_anchor_is_none(self):
        item = _item("id-1")
        result = _build_anchored_concepts([item])

        assert result[0].quote_anchor is None

    def test_one_anchored_concept_per_item(self):
        items = [_item(f"item-{i}") for i in range(4)]
        result = _build_anchored_concepts(items)

        assert len(result) == 4
        ids = [ac.item.title for ac in result]
        assert ids == ["item-0", "item-1", "item-2", "item-3"]

    def test_empty_items_returns_empty_list(self):
        assert _build_anchored_concepts([]) == []


class TestBuildDataframe:
    def test_returns_dataframe_with_one_row_per_example(self):
        examples = [_example("ex-1"), _example("ex-2")]
        df = _build_dataframe(examples)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_inputs_contains_corpus_id(self):
        ex = _example("my-corpus")
        df = _build_dataframe([ex])
        inputs = df.iloc[0]["inputs"]

        assert inputs["corpus_id"] == "my-corpus"

    def test_inputs_items_contain_id_and_description(self):
        items = [_item("it-a", description="Desc A"), _item("it-b", description="Desc B")]
        ex = GroupingCorpusExample(
            corpus_id="corp",
            description="d",
            items=items,
            expected_groups=[["it-a", "it-b"]],
        )
        df = _build_dataframe([ex])
        _items = df.iloc[0]["inputs"]["_items"]

        assert _items == [
            {"id": "it-a", "description": "Desc A"},
            {"id": "it-b", "description": "Desc B"},
        ]

    def test_expectations_contain_expected_groups(self):
        ex = GroupingCorpusExample(
            corpus_id="corp",
            description="d",
            items=[_item("a"), _item("b"), _item("c")],
            expected_groups=[["a", "b"], ["c"]],
        )
        df = _build_dataframe([ex])
        expectations = df.iloc[0]["expectations"]

        assert expectations["expected_groups"] == [["a", "b"], ["c"]]

    def test_tags_are_prefixed_with_corpus_dot(self):
        ex = GroupingCorpusExample(
            corpus_id="corp",
            description="d",
            items=[_item("x")],
            expected_groups=[["x"]],
            tags={"concept_type": "decision", "difficulty": "hard"},
        )
        df = _build_dataframe([ex])
        tags = df.iloc[0]["tags"]

        assert tags == {"corpus.concept_type": "decision", "corpus.difficulty": "hard"}

    def test_empty_examples_produces_empty_dataframe(self):
        df = _build_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestAggregateMetrics:
    def test_extracts_exact_match_and_group_hit_rate(self):
        eval_result = make_eval_result(
            {
                "grouping.exact_match/mean": 0.75,
                "grouping.group_hit_rate/mean": 0.9,
            }
        )
        result = _aggregate_metrics(eval_result)

        assert result == {"exact_match": 0.75, "group_hit_rate": 0.9}

    def test_absent_metrics_are_excluded(self):
        eval_result = make_eval_result(
            {
                "grouping.exact_match/mean": 1.0,
                # group_hit_rate absent
            }
        )
        result = _aggregate_metrics(eval_result)

        assert "exact_match" in result
        assert "group_hit_rate" not in result

    def test_returns_floats(self):
        eval_result = make_eval_result(
            {
                "grouping.exact_match/mean": 1,
                "grouping.group_hit_rate/mean": 0,
            }
        )
        result = _aggregate_metrics(eval_result)

        assert isinstance(result["exact_match"], float)
        assert isinstance(result["group_hit_rate"], float)

    def test_empty_metrics_returns_empty_dict(self):
        eval_result = make_eval_result({})
        assert _aggregate_metrics(eval_result) == {}
