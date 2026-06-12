from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.config.settings import RAGConfig
from seshat.models.api import NodeFilter, SearchResult
from seshat.models.enums import ConceptType, NodeStatus
from seshat.pipeline.extraction.node_retriever import NodeRetriever
from tests.helpers import make_node


def _make_service(search_results=None, kb_nodes=None, neighbour_nodes=None, top_k=3):
    rag_config = RAGConfig(top_k=top_k)

    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=search_results or [])

    kb_store = MagicMock()
    kb_store.get_node = AsyncMock(side_effect=lambda nid: next((n for n in (kb_nodes or []) if str(n.id) == nid), None))
    kb_store.get_neighbours = AsyncMock(return_value=neighbour_nodes or [])

    return NodeRetriever(rag_config=rag_config, kb_store=kb_store, vector_store=vector_store)


class TestNodeRetriever:
    @pytest.mark.asyncio
    async def test_returns_matched_kb_nodes(self):
        candidate = make_node("n2", title="Use Redis")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        service = _make_service(search_results=search_results, kb_nodes=[candidate])

        source = make_node("n1")
        result = await service.retrieve(source)
        assert any(n.id == candidate.id for n in result)

    @pytest.mark.asyncio
    async def test_excludes_source_node_from_results(self):
        source = make_node("n1")
        search_results = [SearchResult(node_id=str(source.id), score=0.99)]
        service = _make_service(search_results=search_results, kb_nodes=[source])

        result = await service.retrieve(source)
        assert not any(n.id == source.id for n in result)

    @pytest.mark.asyncio
    async def test_includes_neighbours(self):
        candidate = make_node("n2", title="Use Redis")
        neighbour = make_node("n3", title="Redis Caching Decision")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        service = _make_service(
            search_results=search_results,
            kb_nodes=[candidate],
            neighbour_nodes=[neighbour],
        )

        source = make_node("n1")
        result = await service.retrieve(source)
        assert any(n.id == neighbour.id for n in result)

    @pytest.mark.asyncio
    async def test_caller_node_filter_overrides_default_status(self):
        service = _make_service()
        source = make_node("n1")
        override = NodeFilter(status=NodeStatus.PENDING_REVIEW)

        await service.retrieve(source, node_filter=override)

        call_kwargs = service._vs.search.call_args.kwargs
        node_filter: NodeFilter = call_kwargs["node_filter"]
        assert node_filter.status == NodeStatus.PENDING_REVIEW

    @pytest.mark.asyncio
    async def test_orphan_vector_result_is_silently_skipped(self):
        # vector store returns a hit, but the KB has no matching node
        orphan_id = str(make_node("orphan").id)
        search_results = [SearchResult(node_id=orphan_id, score=0.9)]
        service = _make_service(search_results=search_results, kb_nodes=[])

        source = make_node("n1")
        result = await service.retrieve(source)

        assert result == []

    @pytest.mark.asyncio
    async def test_exclude_job_id_forwarded_to_vector_search(self):
        service = _make_service()
        source = make_node("n1")

        await service.retrieve(source, exclude_job_id="job-42")

        call_kwargs = service._vs.search.call_args.kwargs
        assert call_kwargs["exclude_job_id"] == "job-42"

    @pytest.mark.asyncio
    async def test_fetch_loop_stops_at_cap_without_fetching_remaining_results(self):
        # top_k=1 → cap=2; three vector hits — only 2 KB fetches should happen
        candidates = [make_node(f"n{i}", title=f"Node {i}") for i in range(2, 5)]
        search_results = [SearchResult(node_id=str(c.id), score=0.9) for c in candidates]
        service = _make_service(search_results=search_results, kb_nodes=candidates, top_k=1)

        await service.retrieve(make_node("n1"))

        assert service._kb.get_node.call_count == 2

    @pytest.mark.asyncio
    async def test_token_budget_stops_fetch_before_top_k_cap(self):
        # each node costs ~9 tokens (title + description); budget of 18 allows 2 nodes
        # top_k=10 → cap=20, so token budget is the binding constraint here
        candidates = [make_node(f"n{i}", title=f"Node {i}") for i in range(2, 6)]
        search_results = [SearchResult(node_id=str(c.id), score=0.9) for c in candidates]

        kb_store = MagicMock()
        kb_store.get_node = AsyncMock(side_effect=lambda nid: next((n for n in candidates if str(n.id) == nid), None))
        kb_store.get_neighbours = AsyncMock(return_value=[])
        service = NodeRetriever(
            rag_config=RAGConfig(top_k=10, max_context_tokens=18),
            kb_store=kb_store,
            vector_store=MagicMock(search=AsyncMock(return_value=search_results)),
        )

        result = await service.retrieve(make_node("n1"))

        assert len(result) == 2
        assert kb_store.get_node.call_count == 2

    @pytest.mark.asyncio
    async def test_cap_limits_neighbour_expansion(self):
        # top_k=1 → cap=2; one vector hit plus three neighbours — only 2 total should be kept
        candidate = make_node("n2", title="Use Redis")
        neighbours = [make_node(f"n{i}", title=f"Neighbour {i}") for i in range(3, 7)]
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        service = _make_service(
            search_results=search_results,
            kb_nodes=[candidate],
            neighbour_nodes=neighbours,
            top_k=1,
        )

        source = make_node("n1")
        result = await service.retrieve(source)

        assert len(result) <= 2  # cap = top_k * 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("override_filter", "expected_type"),
        [
            (None, ConceptType.DECISION),  # default: uses source node type
            (NodeFilter(node_type=ConceptType.RISK), ConceptType.RISK),  # explicit override
            (NodeFilter(node_type=None), None),  # explicit None override
        ],
    )
    async def test_node_type_forwarded_to_vector_search(self, override_filter, expected_type):
        service = _make_service()
        source = make_node("n1", type=ConceptType.DECISION)

        await service.retrieve(source, node_filter=override_filter)

        call_kwargs = service._vs.search.call_args.kwargs
        node_filter = call_kwargs["node_filter"]
        assert node_filter.node_type == expected_type

    @pytest.mark.asyncio
    async def test_no_duplicates_in_result(self):
        candidate = make_node("n2", title="Use Redis")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        # neighbour is the same node as candidate — should not appear twice
        service = _make_service(
            search_results=search_results,
            kb_nodes=[candidate],
            neighbour_nodes=[candidate],
        )

        source = make_node("n1")
        result = await service.retrieve(source)
        ids = [n.id for n in result]
        assert len(ids) == len(set(ids))
