import uuid
from pathlib import Path

from seshat.eval.resolution_corpus_loader import build_kb_nodes_with_slug_map, load_resolution_corpus
from seshat.models.enums import RelationshipType

CORPUS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "eval" / "test_corpus" / "resolution"


class TestLoadResolutionCorpus:
    def test_loads_example(self):
        examples = load_resolution_corpus(CORPUS_DIR)
        assert len(examples) == 1

    def test_example_has_source_and_kb_nodes(self):
        examples = load_resolution_corpus(CORPUS_DIR)
        ex = examples[0]
        assert len(ex.source_nodes) == 1
        assert len(ex.kb_nodes) == 1

    def test_expected_relations_parsed(self):
        examples = load_resolution_corpus(CORPUS_DIR)
        ex = examples[0]
        assert len(ex.expected_relations) == 1
        assert ex.expected_relations[0].rel_type == RelationshipType.AMENDS

    def test_build_kb_nodes_with_slug_map(self):
        examples = load_resolution_corpus(CORPUS_DIR)
        ex = examples[0]
        all_slugs = [n.id for n in ex.source_nodes + ex.kb_nodes]
        kb_nodes, slug_map = build_kb_nodes_with_slug_map(ex)

        for slug in all_slugs:
            assert slug in slug_map
            assert slug in kb_nodes

        for uid in slug_map.values():
            assert isinstance(uid, uuid.UUID)
