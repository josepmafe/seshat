from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock

from seshat.core.config.eval_settings import EvalConfig
from seshat.core.config.settings import RAGConfig
from seshat.core.models.enums import ConceptType
from seshat.eval.models import RetrievalCorpusExample, RetrievalCorpusNode
from seshat.eval.vector_search.runner import VectorSearchEvalRunner

if TYPE_CHECKING:
    from pathlib import Path


def _make_runner(captured_filters: list | None = None) -> VectorSearchEvalRunner:
    """Build a VectorSearchEvalRunner with a mock search engine that captures node_filter kwargs."""
    filters = captured_filters if captured_filters is not None else []

    async def _search(query: str, *, node_filter=None, exclude_job_id=None, score_threshold=None, top_k=None):
        filters.append(node_filter)
        return []

    search_engine = MagicMock()
    search_engine.search = AsyncMock(side_effect=_search)
    search_engine.fingerprint = Mock(return_value="test-fp")

    vs = MagicMock()
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()

    config = Mock(spec=EvalConfig)
    config.vector_search_score_thresholds = {}
    rag_config = RAGConfig()
    return VectorSearchEvalRunner(
        search_engine=search_engine,
        vector_store=vs,
        config=config,
        rag_config=rag_config,
    )


def _make_cross_type_example() -> RetrievalCorpusExample:
    """A corpus example where query_node and candidate_nodes have different types."""
    return RetrievalCorpusExample(
        corpus_id="test_cross_type",
        description="DECISION query, RISK candidates",
        query_node=RetrievalCorpusNode(
            id="decision-1",
            type=ConceptType.DECISION,
            title="Adopt microservices",
            description="We will migrate to a microservices architecture.",
            quote="We decided to migrate to microservices.",
        ),
        candidate_nodes=[
            RetrievalCorpusNode(
                id="risk-1",
                type=ConceptType.RISK,
                title="Service sprawl",
                description="Too many services become hard to manage.",
                quote="Risk of too many services.",
            ),
        ],
        expected_relevant_ids=["risk-1"],
    )


class TestFetchExampleNodeFilter:
    async def test_search_uses_untyped_filter(self, tmp_path: Path) -> None:
        """_fetch_example must pass NodeFilter(node_type=None) so cross-type candidates are searchable."""
        captured_filters: list = []
        runner = _make_runner(captured_filters)

        example = _make_cross_type_example()
        await runner._fetch_example(example)

        assert len(captured_filters) == 1
        captured = captured_filters[0]
        assert captured is not None
        assert captured.node_type is None
