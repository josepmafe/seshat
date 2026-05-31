import pytest

from seshat.config.settings import EvalConfig
from seshat.eval.identification.corpus_loader import load_corpus
from seshat.models.enums import ConceptType


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.identification_corpus_dir)


class TestCorpusLoader:
    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_corpus_examples_have_valid_content(self, examples):
        for ex in examples:
            assert ex.transcript.strip()
            for node in ex.expected_nodes:
                assert node.quote.strip()
                assert isinstance(node.type, ConceptType)


class TestProductionCorpus:
    def test_all_files_load_and_have_valid_content(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.identification_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            assert ex.corpus_id
            assert ex.transcript.strip()
            for node in ex.expected_nodes:
                assert node.quote.strip()
                assert isinstance(node.type, ConceptType)
                assert node.title.strip()
                assert node.description.strip()

    def test_tags_are_parsed(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.identification_corpus_dir)
        tagged = [ex for ex in examples if ex.tags]
        assert tagged, "expected at least one production corpus file to have tags"

    def test_tag_filter_includes_matching(self, eval_corpus: EvalConfig):
        all_examples = load_corpus(eval_corpus.identification_corpus_dir)
        tiers = {ex.tags.get("tier") for ex in all_examples if "tier" in ex.tags}
        assert tiers, "expected at least one example with a 'tier' tag"

        tier = next(iter(tiers))
        filtered = load_corpus(eval_corpus.identification_corpus_dir, tag_filter={"tier": tier})
        assert all(ex.tags.get("tier") == tier for ex in filtered)
        assert len(filtered) < len(all_examples)

    def test_tag_filter_excludes_non_matching(self, eval_corpus: EvalConfig):
        filtered = load_corpus(eval_corpus.identification_corpus_dir, tag_filter={"tier": "__nonexistent__"})
        assert filtered == []

    def test_tag_filter_none_returns_all(self, eval_corpus: EvalConfig):
        all_examples = load_corpus(eval_corpus.identification_corpus_dir)
        filtered = load_corpus(eval_corpus.identification_corpus_dir, tag_filter=None)
        assert len(filtered) == len(all_examples)
