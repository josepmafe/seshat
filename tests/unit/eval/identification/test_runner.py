from __future__ import annotations

import pandas as pd
import pytest

from seshat.eval.identification.runner import _aggregate_metrics, _build_dataframe, _first_quote, _slim_node
from seshat.eval.models import IdentificationCorpusExample
from seshat.models.enums import ConceptType
from tests.helpers import make_node
from tests.unit.eval.helpers import make_eval_result
from tests.unit.eval.identification.helpers import corpus_node

# ── _slim_node ────────────────────────────────────────────────────────────────


def test_slim_node_keeps_required_fields():
    n = {"type": "decision", "title": "Use Postgres", "description": "We chose Postgres."}
    result = _slim_node(n)
    assert result == {"type": "decision", "title": "Use Postgres", "description": "We chose Postgres."}


def test_slim_node_includes_confidence_when_present():
    n = {"type": "decision", "title": "T", "description": "D", "confidence": 0.85}
    result = _slim_node(n)
    assert result["confidence"] == 0.85


def test_slim_node_omits_confidence_when_absent():
    n = {"type": "decision", "title": "T", "description": "D"}
    result = _slim_node(n)
    assert "confidence" not in result


def test_slim_node_strips_extra_keys():
    n = {
        "type": "decision",
        "title": "T",
        "description": "D",
        "quote": "Some quote",
        "assignee": "Alice",
        "rationale": "Because reasons",
    }
    result = _slim_node(n)
    assert set(result.keys()) == {"type", "title", "description"}


# ── _build_dataframe ──────────────────────────────────────────────────────────


def _make_example(
    corpus_id: str = "ex_001",
    transcript: str = "Meeting transcript.",
    nodes: list | None = None,
    tags: dict | None = None,
) -> IdentificationCorpusExample:
    return IdentificationCorpusExample(
        corpus_id=corpus_id,
        transcript=transcript,
        expected_nodes=nodes or [corpus_node("q", ConceptType.DECISION)],
        tags=tags or {},
    )


def test_build_dataframe_one_row_per_example():
    examples = [_make_example("id1"), _make_example("id2"), _make_example("id3")]
    df = _build_dataframe(examples)
    assert len(df) == 3


def test_build_dataframe_has_required_columns():
    df = _build_dataframe([_make_example()])
    assert set(df.columns) == {"inputs", "expectations", "tags"}


def test_build_dataframe_inputs_shape():
    ex = _make_example(corpus_id="abc", transcript="hello world")
    df = _build_dataframe([ex])
    inputs = df.iloc[0]["inputs"]
    assert inputs["corpus_id"] == "abc"
    assert inputs["transcript"] == "hello world"


def test_build_dataframe_expectations_contain_expected_nodes():
    node = corpus_node(quote="we decided", ctype=ConceptType.RISK)
    ex = _make_example(nodes=[node])
    df = _build_dataframe([ex])
    expected_nodes = df.iloc[0]["expectations"]["expected_nodes"]
    assert isinstance(expected_nodes, list)
    assert len(expected_nodes) == 1
    assert expected_nodes[0]["type"] == ConceptType.RISK


def test_build_dataframe_tags_prefixed_with_corpus():
    ex = _make_example(tags={"difficulty": "hard", "source": "zoom"})
    df = _build_dataframe([ex])
    tags = df.iloc[0]["tags"]
    assert tags["corpus.difficulty"] == "hard"
    assert tags["corpus.source"] == "zoom"


def test_build_dataframe_tags_values_are_strings():
    ex = _make_example(tags={"count": 42})
    df = _build_dataframe([ex])
    tags = df.iloc[0]["tags"]
    assert tags["corpus.count"] == "42"


def test_build_dataframe_empty_input():
    df = _build_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── _aggregate_metrics ────────────────────────────────────────────────────────


def test_aggregate_metrics_extracts_precision_recall_spurious():
    ctype = ConceptType.DECISION
    result = _aggregate_metrics(
        make_eval_result(
            {
                f"{ctype}.precision/mean": 0.9,
                f"{ctype}.recall/mean": 0.8,
                f"{ctype}.spurious_rate/mean": 0.1,
            }
        )
    )
    assert result[f"{ctype}.precision"] == pytest.approx(0.9)
    assert result[f"{ctype}.recall"] == pytest.approx(0.8)
    assert result[f"{ctype}.spurious_rate"] == pytest.approx(0.1)


def test_aggregate_metrics_omits_key_when_none():
    ctype = ConceptType.DECISION
    result = _aggregate_metrics(
        make_eval_result(
            {
                f"{ctype}.precision/mean": 0.9,
                # recall and spurious_rate absent
            }
        )
    )
    assert f"{ctype}.precision" in result
    assert f"{ctype}.recall" not in result
    assert f"{ctype}.spurious_rate" not in result


def test_aggregate_metrics_converts_to_float():
    ctype = ConceptType.RISK
    # Pass an int-like value to confirm float() conversion happens
    result = _aggregate_metrics(make_eval_result({f"{ctype}.precision/mean": 1}))
    value = result[f"{ctype}.precision"]
    assert isinstance(value, float)
    assert value == pytest.approx(1.0)


def test_aggregate_metrics_covers_all_concept_types():
    metrics = {}
    for ctype in ConceptType:
        metrics[f"{ctype}.precision/mean"] = 0.5

    result = _aggregate_metrics(make_eval_result(metrics))

    for ctype in ConceptType:
        assert f"{ctype}.precision" in result


def test_aggregate_metrics_empty_metrics_returns_empty_dict():
    result = _aggregate_metrics(make_eval_result({}))
    assert result == {}


# ── _first_quote ──────────────────────────────────────────────────────────────


def test_first_quote_returns_substring_from_anchor():
    transcript = "We decided to adopt PostgreSQL for the project."
    quote = "adopt PostgreSQL"
    node = make_node(quote=quote, transcript=transcript)
    result = _first_quote(node, transcript)
    assert result == quote


def test_first_quote_uses_first_anchor_when_multiple():
    transcript = "Alpha Beta Gamma Delta"
    # Provide two anchors; only first should be used
    from seshat.models.quote_anchor import QuoteAnchor

    anchors = [
        QuoteAnchor(transcript_file="t.txt", char_start=0, char_end=5),  # "Alpha"
        QuoteAnchor(transcript_file="t.txt", char_start=6, char_end=10),  # "Beta"
    ]
    node = make_node(quote_anchors=anchors)
    result = _first_quote(node, transcript)
    assert result == "Alpha"


def test_first_quote_returns_none_when_no_anchors():

    node = make_node(quote_anchors=[])
    result = _first_quote(node, "some transcript text")
    assert result is None
