from __future__ import annotations

from types import SimpleNamespace

from seshat.eval.corpus_tags import corpus_tag_summary, matches_tags

# --- matches_tags ---


def test_matches_tags_all_scalar_filters_match():
    tags = {"type": "decision", "source": "meeting"}
    assert matches_tags(tags, {"type": "decision", "source": "meeting"}) is True


def test_matches_tags_scalar_filter_mismatch():
    tags = {"type": "decision"}
    assert matches_tags(tags, {"type": "action_item"}) is False


def test_matches_tags_list_filter_subset_of_tag_value():
    tags = {"labels": ["a", "b", "c"]}
    assert matches_tags(tags, {"labels": ["a", "b"]}) is True


def test_matches_tags_list_filter_item_not_in_tag_value():
    tags = {"labels": ["a", "b"]}
    assert matches_tags(tags, {"labels": ["a", "z"]}) is False


def test_matches_tags_empty_filter_always_true():
    tags = {"type": "decision", "source": "meeting"}
    assert matches_tags(tags, {}) is True


def test_matches_tags_missing_key_returns_false():
    tags = {"source": "meeting"}
    assert matches_tags(tags, {"type": "decision"}) is False


def test_matches_tags_list_filter_against_non_list_tag_value():
    # Tag value is a scalar, but filter expects a list — should not match
    tags = {"labels": "a"}
    assert matches_tags(tags, {"labels": ["a"]}) is False


# --- corpus_tag_summary ---


def _ex(tags: dict) -> SimpleNamespace:
    return SimpleNamespace(tags=tags)


def test_corpus_tag_summary_single_example():
    examples = [_ex({"type": "decision", "source": "meeting"})]
    result = corpus_tag_summary(examples)
    assert result == {"corpus.type": "decision", "corpus.source": "meeting"}


def test_corpus_tag_summary_merges_and_sorts_values_across_examples():
    examples = [
        _ex({"type": "decision"}),
        _ex({"type": "action_item"}),
        _ex({"type": "decision"}),
    ]
    result = corpus_tag_summary(examples)
    assert result == {"corpus.type": "action_item,decision"}


def test_corpus_tag_summary_default_blacklist_excludes_detail():
    examples = [_ex({"type": "decision", "detail": "some detail"})]
    result = corpus_tag_summary(examples)
    assert "corpus.detail" not in result
    assert "corpus.type" in result


def test_corpus_tag_summary_custom_blacklist_overrides_default():
    # "detail" is no longer blacklisted; "source" is blacklisted instead
    examples = [_ex({"detail": "x", "source": "meeting", "type": "risk"})]
    result = corpus_tag_summary(examples, tags_blacklist=("source",))
    assert "corpus.detail" in result
    assert "corpus.source" not in result
    assert "corpus.type" in result


def test_corpus_tag_summary_empty_examples():
    assert corpus_tag_summary([]) == {}
