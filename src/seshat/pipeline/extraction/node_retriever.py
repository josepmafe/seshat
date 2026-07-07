from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.enums import GraphDirection
from seshat.core.utils.log import get_logger
from seshat.core.utils.retry import async_retry
from seshat.core.utils.tokens import count_tokens

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.core.config.settings import RAGConfig
    from seshat.core.models.api_graph import SearchResult
    from seshat.core.models.nodes import KBNode
    from seshat.repositories.node_repository import NodeRepository

logger = get_logger(__name__)


class Reranker(Protocol):
    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]: ...


class NodeRetriever:
    def __init__(
        self,
        rag_config: RAGConfig,
        node_repo: NodeRepository,
        reranker: Reranker | None = None,
    ) -> None:
        self._config = rag_config
        self._repo = node_repo
        self._reranker = reranker

    @property
    def max_concurrent_retrievals(self) -> int:
        return self._config.max_concurrent_retrievals

    @property
    def node_retrieval_cap(self) -> int:
        return self._config.top_k * 2  # doubled to leave room for neighbour-expansion below

    async def retrieve(
        self,
        node: KBNode,
        *,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
    ) -> list[KBNode]:
        # Vector store only holds approved+current nodes — status/state are enforced
        # by only upserting on approval and deleting on archival (TODO: implement delete hook).
        filter_kwargs: dict = {"node_type": node.type}
        if node_filter is not None:
            filter_kwargs.update(node_filter.model_dump(exclude_unset=True))

        query = _build_vector_search_query(node)
        logger.debug("Retrieving targets for node id=%s type=%s", node.id, node.type.value)

        results = await self._vector_search(
            query, node_filter=NodeFilter(**filter_kwargs), exclude_job_id=exclude_job_id
        )
        logger.info("Vector search returned %d raw results for node id=%s", len(results), node.id)

        if self._reranker is not None:
            results = await self._reranker.rerank(query, results)

        budget = _ContextBudget(self._config.max_context_tokens)
        seen: dict[UUID, KBNode] = {}

        # TOCONSIDER: retrieve direct hits in parallel: faster but potential wasted KB calls on nodes we'd discard
        await self._fetch_direct_hits(seen, results, node.id, budget)
        # TOCONSIDER: retrieved neighbours in parallel, re-rerank them and take top-k.
        await self._expand_with_neighbours(seen, results, node.id, budget)

        targets = list(seen.values())
        logger.info("target retrieval done: %d targets for node id=%s", len(targets), node.id)
        return targets

    async def _fetch_direct_hits(
        self,
        seen: dict[UUID, KBNode],
        results: list[SearchResult],
        node_id: UUID,
        budget: _ContextBudget,
    ) -> None:
        # Sequential fetch to allow early exit on node cap or budget;
        # parallel gather would waste KB calls on nodes we'd discard.
        for result in results:
            if result.node_id in seen:
                logger.warning("Duplicate node id=%s found in vector search results; skipping", result.node_id)
                continue

            if result.node_id == node_id:
                continue

            if len(seen) >= self.node_retrieval_cap or budget.exhausted:
                break

            kb_node = await self._repo.get_node(result.node_id)
            if kb_node is None:
                logger.warning("Node id=%s found in vector search but missing from KB store", result.node_id)
                continue

            if not budget.consume(kb_node):
                logger.debug("Context budget exhausted; stopping at %d direct hits", len(seen))
                break

            seen[result.node_id] = kb_node

    async def _expand_with_neighbours(
        self,
        seen: dict[UUID, KBNode],
        results: list[SearchResult],
        node_id: UUID,
        budget: _ContextBudget,
    ) -> None:
        for result in results:
            if len(seen) >= self.node_retrieval_cap or budget.exhausted:
                break

            if result.node_id not in seen:
                continue

            neighbours = await self._repo.get_neighbours(
                result.node_id, rel_types=self._config.traversal_rel_types, direction=GraphDirection.BOTH
            )
            for neighbour in neighbours:
                # we are not checking if budget is already exhausted, since the neighbours
                # are already in memory and `consume` allows slight overage before rejecting
                if len(seen) >= self.node_retrieval_cap:
                    break

                if neighbour.id == node_id or neighbour.id in seen:
                    continue

                if not budget.consume(neighbour):
                    # check if there is any "cheaper" neighbour that would fit in the remaining budget,
                    # instead of just skipping all remaining neighbours once we hit the first expensive one
                    continue

                seen[neighbour.id] = neighbour

    # Retry kept here (not in the vector store) because retryable exceptions are
    # provider-specific (httpx, openai) and don't belong in the store abstraction
    @async_retry()
    async def _vector_search(
        self, query: str, node_filter: NodeFilter, *, exclude_job_id: str | None
    ) -> list[SearchResult]:
        return await self._repo.search(
            query,
            top_k=self._config.top_k,
            node_filter=node_filter,
            exclude_job_id=exclude_job_id,
            score_threshold=self._config.min_similarity_score,
            mode=self._config.search_mode,
        )


def _build_vector_search_query(node: KBNode) -> str:
    return f"{node.title} {node.description}"


class _ContextBudget:
    _OVERAGE = 1.1  # allow up to 10% over the soft cap before rejecting a node

    def __init__(self, max_tokens: int) -> None:
        self._soft_limit = max_tokens
        self._hard_limit = int(max_tokens * self._OVERAGE)
        self._used = 0

    @property
    def exhausted(self) -> bool:
        """True once the soft cap is reached — used to skip further KB fetches."""
        return self._used >= self._soft_limit

    def consume(self, node: KBNode) -> bool:
        """Deduct token cost of node. Returns False (and does not deduct) if it would exceed the hard limit."""
        cost = count_tokens(_build_vector_search_query(node))
        if self._used + cost > self._hard_limit:
            return False
        self._used += cost
        return True
