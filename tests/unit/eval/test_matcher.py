from seshat.eval.matcher import QUOTE_MATCH_THRESHOLD, MatchMethod, match_nodes
from seshat.eval.models import IdentificationCorpusNode
from seshat.models.enums import ConceptType, NodeStatus
from tests.helpers import make_node

TRANSCRIPT = (
    "We decided to use PostgreSQL for all operational data. "
    "There is a risk that replication lag could affect reads. "
    "Alice will benchmark the replica setup by Friday."
)


def _corpus_node(quote: str, ctype: ConceptType, description: str = "A description.") -> IdentificationCorpusNode:
    return IdentificationCorpusNode(quote=quote, type=ctype, title="T", description=description)


class TestMatchNodes:
    def test_exact_match(self):
        quote = "We decided to use PostgreSQL for all operational data."
        predicted = [make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.DECISION)]
        expected = [_corpus_node(quote, ConceptType.DECISION)]
        result = match_nodes(TRANSCRIPT, expected, predicted)
        assert len(result.matched) == 1
        assert result.matched[0].match_score >= QUOTE_MATCH_THRESHOLD
        assert len(result.missed) == 0
        assert len(result.spurious) == 0

    def test_type_mismatch_is_not_a_match(self):
        quote = "We decided to use PostgreSQL for all operational data."
        predicted = [make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.RISK)]
        expected = [_corpus_node(quote, ConceptType.DECISION)]
        result = match_nodes(TRANSCRIPT, expected, predicted)
        assert len(result.matched) == 0
        assert len(result.missed) == 1
        assert len(result.spurious) == 1

    def test_spurious_node(self):
        quote = "Alice will benchmark the replica setup by Friday."
        predicted = [make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.ACTION_ITEM)]
        result = match_nodes(TRANSCRIPT, [], predicted)
        assert len(result.spurious) == 1
        assert len(result.matched) == 0

    def test_missed_node(self):
        expected = [_corpus_node("There is a risk that replication lag could affect reads.", ConceptType.RISK)]
        result = match_nodes(TRANSCRIPT, expected, [])
        assert len(result.missed) == 1
        assert result.missed[0].quote == expected[0].quote


class TestTitleFallback:
    def test_semantically_matching_node_scores_above_threshold(self):
        """A node with no anchors but matching title and description should match."""
        node = make_node(
            title="Use PostgreSQL for operational data",
            description="The team chose PostgreSQL for all operational data storage.",
            type=ConceptType.DECISION,
            status=NodeStatus.PENDING_REVIEW,
            quote_anchors=[],
        )
        expected = [
            IdentificationCorpusNode(
                quote="We decided to use PostgreSQL for all operational data.",
                type=ConceptType.DECISION,
                title="Use PostgreSQL for operational data",
                description="The team chose PostgreSQL for all operational data storage.",
            )
        ]
        result = match_nodes(TRANSCRIPT, expected, [node])
        assert len(result.matched) == 1
        assert result.matched[0].matched_by == MatchMethod.TITLE_FALLBACK
        assert result.matched[0].match_score >= QUOTE_MATCH_THRESHOLD

    def test_semantically_unrelated_node_scores_below_threshold(self):
        """A node with no anchors and unrelated title+description should not match."""
        node = make_node(
            title="Migrate to Kubernetes",
            description="The team decided to move all services to Kubernetes for orchestration.",
            type=ConceptType.DECISION,
            status=NodeStatus.PENDING_REVIEW,
            quote_anchors=[],
        )
        expected = [
            IdentificationCorpusNode(
                quote="We decided to use PostgreSQL for all operational data.",
                type=ConceptType.DECISION,
                title="Use PostgreSQL for operational data",
                description="The team chose PostgreSQL for all operational data storage.",
            )
        ]
        result = match_nodes(TRANSCRIPT, expected, [node])
        assert len(result.matched) == 0
        assert len(result.missed) == 1
        assert len(result.spurious) == 1
