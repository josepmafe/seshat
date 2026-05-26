from pathlib import Path

from seshat.eval.retrieval_corpus_loader import load_retrieval_corpus

CORPUS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "eval" / "test_corpus" / "retrieval"


class TestLoadRetrievalCorpus:
    def test_loads_example(self):
        examples = load_retrieval_corpus(CORPUS_DIR)
        assert len(examples) == 1

    def test_query_node_parsed(self):
        examples = load_retrieval_corpus(CORPUS_DIR)
        ex = examples[0]
        assert ex.query_node.id == "delay-deadline"
        assert ex.query_node.title == "Delay Q3 deadline by two weeks"

    def test_candidates_parsed(self):
        examples = load_retrieval_corpus(CORPUS_DIR)
        ex = examples[0]
        assert len(ex.candidate_nodes) == 2

    def test_expected_relevant_ids(self):
        examples = load_retrieval_corpus(CORPUS_DIR)
        ex = examples[0]
        assert ex.expected_relevant_ids == ["scope-creep-risk"]
