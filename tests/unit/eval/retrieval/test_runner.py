from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from seshat.config.eval_settings import EvalConfig
from seshat.eval.models import RetrievalCorpusExample, RetrievalCorpusNode
from seshat.eval.retrieval.runner import RetrievalEvalRunner
from seshat.models.enums import ConceptType
from seshat.vector_store.base_store import AbstractVectorStore

if TYPE_CHECKING:
    from pathlib import Path

    from seshat.models.api import NodeFilter, SearchResult


class _CapturingVectorStore(AbstractVectorStore):
    """Records the NodeFilter and query string passed to each search() call."""

    def __init__(self) -> None:
        self.captured_filters: list[NodeFilter | None] = []
        self.captured_queries: list[str] = []

    @staticmethod
    def get_supported_filter_fields() -> frozenset[str]:
        return frozenset({"node_type"})

    async def upsert(self, node_id: str, text: str, metadata: dict) -> None:
        pass

    async def search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        self.captured_filters.append(node_filter)
        self.captured_queries.append(query)
        return []

    async def delete(self, node_id: str) -> None:
        pass


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
    @pytest.mark.asyncio
    async def test_search_uses_untyped_filter(self, tmp_path: Path) -> None:
        """_fetch_example must pass NodeFilter(node_type=None) so cross-type candidates are searchable."""
        vs = _CapturingVectorStore()
        config = Mock(spec=EvalConfig)
        runner = RetrievalEvalRunner(vector_store=vs, config=config)

        example = _make_cross_type_example()
        await runner._fetch_example(example)

        assert len(vs.captured_filters) == 1
        captured = vs.captured_filters[0]
        assert captured is not None
        assert captured.node_type is None

    @pytest.mark.asyncio
    async def test_query_includes_truncated_quote(self) -> None:
        """_fetch_example must include the first 80 chars of the query node quote, matching NodeRetriever."""
        vs = _CapturingVectorStore()
        config = Mock(spec=EvalConfig)
        runner = RetrievalEvalRunner(vector_store=vs, config=config)

        example = _make_cross_type_example()
        await runner._fetch_example(example)

        assert len(vs.captured_queries) == 1
        query = vs.captured_queries[0]
        expected_quote_fragment = example.query_node.quote[:80]
        assert expected_quote_fragment in query
