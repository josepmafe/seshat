import uuid

import pytest

from seshat.config.settings import EvalConfig
from seshat.eval.resolution.corpus_loader import build_kb_nodes, load_corpus


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.resolution_corpus_dir)


class TestCorpusLoader:
    def test_loads_example(self, examples):
        assert len(examples) > 0

    def test_build_kb_nodes_with_slug_map(self, examples):
        ex = examples[0]
        all_slugs = [n.id for n in ex.source_nodes + ex.kb_nodes]
        kb_nodes, slug_map = build_kb_nodes(ex)

        for slug in all_slugs:
            assert slug in slug_map
            assert slug in kb_nodes

        for uid in slug_map.values():
            assert isinstance(uid, uuid.UUID)


class TestProductionCorpus:
    def test_all_files_load_and_slugs_resolve(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.resolution_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            _, slug_map = build_kb_nodes(ex)
            all_slugs = set(slug_map.keys())
            for r in ex.expected_relations:
                assert r.source in all_slugs, f"{ex.corpus_id}: unknown source slug {r.source!r}"
                assert r.target in all_slugs, f"{ex.corpus_id}: unknown target slug {r.target!r}"

    def test_tags_are_parsed(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.resolution_corpus_dir)
        tagged = [ex for ex in examples if ex.tags]
        assert tagged, "expected at least one production corpus file to have tags"

    def test_tag_filter_includes_matching(self, eval_corpus: EvalConfig):
        all_examples = load_corpus(eval_corpus.resolution_corpus_dir)
        tiers = {ex.tags.get("tier") for ex in all_examples if "tier" in ex.tags}
        assert tiers, "expected at least one example with a 'tier' tag"

        tier = next(iter(tiers))
        filtered = load_corpus(eval_corpus.resolution_corpus_dir, tag_filter={"tier": tier})
        assert all(ex.tags.get("tier") == tier for ex in filtered)
        assert len(filtered) < len(all_examples)

    def test_tag_filter_excludes_non_matching(self, eval_corpus: EvalConfig):
        filtered = load_corpus(eval_corpus.resolution_corpus_dir, tag_filter={"tier": "__nonexistent__"})
        assert filtered == []

    def test_tag_filter_none_returns_all(self, eval_corpus: EvalConfig):
        all_examples = load_corpus(eval_corpus.resolution_corpus_dir)
        filtered = load_corpus(eval_corpus.resolution_corpus_dir, tag_filter=None)
        assert len(filtered) == len(all_examples)
