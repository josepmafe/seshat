from pathlib import Path

from seshat.eval.identification_corpus_loader import load_identification_corpus
from seshat.models.enums import ConceptType

CORPUS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "eval" / "corpus" / "identification"


class TestCorpusLoader:
    def test_load_all_files(self):
        examples = load_identification_corpus(CORPUS_DIR)
        assert len(examples) == 10

    def test_corpus_example_has_transcript(self):
        examples = load_identification_corpus(CORPUS_DIR)
        for ex in examples:
            assert ex.transcript.strip()

    def test_corpus_nodes_have_quote_and_type(self):
        examples = load_identification_corpus(CORPUS_DIR)
        for ex in examples:
            for node in ex.expected_nodes:
                assert node.quote.strip()
                assert isinstance(node.type, ConceptType)
