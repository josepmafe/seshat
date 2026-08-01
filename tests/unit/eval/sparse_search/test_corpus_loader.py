import pytest

from seshat.core.config.eval_settings import EvalConfig
from seshat.core.models.enums import EvalHarness
from seshat.eval.sparse_search.corpus_loader import build_kb_nodes, load_corpus
from tests.unit.eval.conftest import TagFilterContractTests


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.corpus_dir(EvalHarness.SPARSE_SEARCH))


class TestCorpusLoader:
    def test_loads_example(self, examples):
        assert len(examples) > 0


class TestProductionCorpus(TagFilterContractTests):
    load_corpus = staticmethod(load_corpus)
    harness = EvalHarness.SPARSE_SEARCH
    tag_key = "tier"

    def test_all_files_load_and_ids_resolve(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.corpus_dir(EvalHarness.SPARSE_SEARCH))
        assert len(examples) > 0

        for ex in examples:
            _, _, slug_map = build_kb_nodes(ex)
            candidate_ids = {cn.id for cn in ex.candidate_nodes}
            for rel_id in ex.expected_relevant_ids:
                assert rel_id in candidate_ids, f"{ex.corpus_id}: unknown relevant id {rel_id!r}"
            assert ex.query_node.id in slug_map, f"{ex.corpus_id}: query node id missing from slug_map"

    def test_includes_sparse_saves_cases(self, eval_corpus: EvalConfig):
        """sparse_search's corpus must include the sparse_saves=true cases — those are exactly the
        cases this leg is uniquely suited to win (semantic-only retrieval misses them by design;
        see docs/superpowers/specs/2026-07-31-eval-harness-expansion-design.md §2)."""
        examples = load_corpus(eval_corpus.corpus_dir(EvalHarness.SPARSE_SEARCH))
        sparse_saves = [ex.corpus_id for ex in examples if ex.tags.get("sparse_saves") is True]
        assert len(sparse_saves) == 3, f"expected 3 sparse_saves=true cases, found: {sparse_saves}"
