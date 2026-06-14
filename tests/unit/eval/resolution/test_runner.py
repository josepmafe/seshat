import pandas as pd

from seshat.eval.models import ResolutionCorpusExample, ResolutionCorpusNode, ResolutionCorpusRelation
from seshat.eval.resolution.runner import _aggregate_metrics, _build_dataframe, _is_same_type, _slim_node
from seshat.models.enums import ConceptType, RelationshipType
from tests.unit.eval.helpers import make_eval_result


def _node(slug: str, concept_type: ConceptType, title: str = "t", description: str = "d") -> ResolutionCorpusNode:
    return ResolutionCorpusNode(
        id=slug,
        type=concept_type,
        title=title,
        description=description,
        quote="q",
    )


def _example(source_type: ConceptType, kb_type: ConceptType) -> ResolutionCorpusExample:
    return ResolutionCorpusExample(
        corpus_id="test",
        description="test",
        source_nodes=[_node("src", source_type)],
        kb_nodes=[_node("kb", kb_type)],
        expected_relations=[],
    )


class TestIsSameType:
    def test_same_type_returns_true(self):
        assert _is_same_type(_example(ConceptType.DECISION, ConceptType.DECISION)) is True

    def test_different_types_returns_false(self):
        assert _is_same_type(_example(ConceptType.DECISION, ConceptType.RISK)) is False


class TestSlimNode:
    def test_fields_mapped_correctly(self):
        node = _node("my-slug", ConceptType.RISK, title="Risk title", description="Some risk")
        result = _slim_node(node)

        assert result == {
            "id": "my-slug",
            "type": "risk",
            "title": "Risk title",
            "description": "Some risk",
        }

    def test_type_value_is_string(self):
        node = _node("s", ConceptType.ACTION_ITEM)
        assert isinstance(_slim_node(node)["type"], str)
        assert _slim_node(node)["type"] == "action_item"


class TestBuildDataframe:
    def _make_example(self) -> ResolutionCorpusExample:
        return ResolutionCorpusExample(
            corpus_id="corp-1",
            description="desc",
            source_nodes=[_node("src-a", ConceptType.DECISION, title="Dec A", description="Decision A")],
            kb_nodes=[_node("kb-b", ConceptType.RISK, title="Risk B", description="Risk B desc")],
            expected_relations=[
                ResolutionCorpusRelation(source="src-a", target="kb-b", rel_type=RelationshipType.MITIGATES)
            ],
            tags={"tier": "basic", "type": "cross"},
        )

    def test_returns_dataframe_with_one_row_per_example(self):
        ex1 = self._make_example()
        ex2 = ResolutionCorpusExample(
            corpus_id="corp-2",
            description="desc2",
            source_nodes=[_node("s", ConceptType.RISK)],
            kb_nodes=[_node("k", ConceptType.RISK)],
            expected_relations=[],
        )
        df = _build_dataframe([ex1, ex2])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_inputs_contains_corpus_id_and_slimmed_nodes(self):
        ex = self._make_example()
        df = _build_dataframe([ex])
        row = df.iloc[0]
        inputs = row["inputs"]

        assert inputs["corpus_id"] == "corp-1"
        assert inputs["_source_nodes"] == [
            {"id": "src-a", "type": "decision", "title": "Dec A", "description": "Decision A"}
        ]
        assert inputs["_kb_nodes"] == [{"id": "kb-b", "type": "risk", "title": "Risk B", "description": "Risk B desc"}]

    def test_expectations_contain_relations_and_slug_to_type(self):
        ex = self._make_example()
        df = _build_dataframe([ex])
        expectations = df.iloc[0]["expectations"]

        assert expectations["expected_relations"] == [{"source": "src-a", "target": "kb-b", "rel_type": "mitigates"}]
        assert expectations["slug_to_type"] == {"src-a": "decision", "kb-b": "risk"}

    def test_tags_are_prefixed_with_corpus_dot(self):
        ex = self._make_example()
        df = _build_dataframe([ex])
        tags = df.iloc[0]["tags"]

        assert tags == {"corpus.tier": "basic", "corpus.type": "cross"}

    def test_empty_examples_produces_empty_dataframe(self):
        df = _build_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestAggregateMetrics:
    def test_extracts_precision_and_recall_for_present_types(self):
        eval_result = make_eval_result(
            {
                "decision.precision/mean": 0.9,
                "decision.recall/mean": 0.8,
                "risk.precision/mean": 0.7,
                "risk.recall/mean": 0.6,
            }
        )
        result = _aggregate_metrics(eval_result)

        assert result["decision.precision"] == 0.9
        assert result["decision.recall"] == 0.8
        assert result["risk.precision"] == 0.7
        assert result["risk.recall"] == 0.6

    def test_absent_metrics_are_excluded(self):
        eval_result = make_eval_result(
            {
                "decision.precision/mean": 1.0,
                # no recall for decision, no risk metrics at all
            }
        )
        result = _aggregate_metrics(eval_result)

        assert "decision.precision" in result
        assert "decision.recall" not in result
        assert "risk.precision" not in result

    def test_returns_floats(self):
        eval_result = make_eval_result({"decision.precision/mean": 1})
        result = _aggregate_metrics(eval_result)
        assert isinstance(result["decision.precision"], float)

    def test_empty_metrics_returns_empty_dict(self):
        eval_result = make_eval_result({})
        assert _aggregate_metrics(eval_result) == {}
