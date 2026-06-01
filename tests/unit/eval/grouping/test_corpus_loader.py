import pytest

from seshat.config.settings import EvalConfig
from seshat.eval.grouping.corpus_loader import load_corpus


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.grouping_corpus_dir)


class TestCorpusLoader:
    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_corpus_examples_have_valid_content(self, examples):
        for ex in examples:
            assert ex.corpus_id
            assert ex.description.strip()
            assert len(ex.items) >= 1
            for item in ex.items:
                assert item.id
                assert item.title.strip()
                assert item.description.strip()

    def test_expected_groups_cover_all_items(self, examples):
        for ex in examples:
            all_item_ids = {item.id for item in ex.items}
            grouped_ids = {item_id for group in ex.expected_groups for item_id in group}
            assert grouped_ids == all_item_ids, f"{ex.corpus_id}: expected_groups do not cover all items"


class TestProductionCorpus:
    def test_all_files_load_and_have_valid_content(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.grouping_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            assert ex.corpus_id
            assert ex.description.strip()
            for item in ex.items:
                assert item.id
                assert item.title.strip()

    def test_tags_are_parsed(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.grouping_corpus_dir)
        tagged = [ex for ex in examples if ex.tags]
        assert tagged, "expected at least one production corpus file to have tags"

    def test_tag_filter_includes_matching(self, eval_corpus: EvalConfig):
        all_examples = load_corpus(eval_corpus.grouping_corpus_dir)
        concept_types = {ex.tags.get("concept_type") for ex in all_examples if "concept_type" in ex.tags}
        assert concept_types

        ct = next(iter(concept_types))
        filtered = load_corpus(eval_corpus.grouping_corpus_dir, tag_filter={"concept_type": ct})
        assert all(ex.tags.get("concept_type") == ct for ex in filtered)

    def test_tag_filter_excludes_non_matching(self, eval_corpus: EvalConfig):
        filtered = load_corpus(eval_corpus.grouping_corpus_dir, tag_filter={"concept_type": "__nonexistent__"})
        assert filtered == []

    def test_tag_filter_none_returns_all(self, eval_corpus: EvalConfig):
        all_examples = load_corpus(eval_corpus.grouping_corpus_dir)
        filtered = load_corpus(eval_corpus.grouping_corpus_dir, tag_filter=None)
        assert len(filtered) == len(all_examples)
