import pytest

from seshat.core.config.eval_settings import EvalConfig
from seshat.eval.vector_search.corpus_loader import build_kb_nodes, load_corpus
from tests.unit.eval.conftest import TagFilterContractTests


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.vector_search_corpus_dir)


class TestCorpusLoader:
    def test_loads_example(self, examples):
        assert len(examples) > 0


class TestProductionCorpus(TagFilterContractTests):
    load_corpus = staticmethod(load_corpus)
    corpus_dir_attr = "vector_search_corpus_dir"
    tag_key = "tier"

    def test_all_files_load_and_ids_resolve(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.vector_search_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            _, _, slug_map = build_kb_nodes(ex)
            candidate_ids = {cn.id for cn in ex.candidate_nodes}
            for rel_id in ex.expected_relevant_ids:
                assert rel_id in candidate_ids, f"{ex.corpus_id}: unknown relevant id {rel_id!r}"
            assert ex.query_node.id in slug_map, f"{ex.corpus_id}: query node id missing from slug_map"

    def test_no_sparse_saves_cases(self, eval_corpus: EvalConfig):
        """vector_search's corpus must exclude sparse_saves=true cases — those are constructed so a
        semantic-only retriever misses them by design; including them would misattribute a by-design
        miss as a regression (see docs/superpowers/specs/2026-07-31-eval-harness-expansion-design.md §1)."""
        examples = load_corpus(eval_corpus.vector_search_corpus_dir)
        sparse_saves = [ex.corpus_id for ex in examples if ex.tags.get("sparse_saves") is True]
        assert sparse_saves == [], f"unexpected sparse_saves=true cases in vector_search corpus: {sparse_saves}"
