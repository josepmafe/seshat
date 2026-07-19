from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

from seshat.app.agents.keyword_extraction import KeywordAgent
from seshat.app.agents.multi_query import MultiQueryAgent
from seshat.core.models.api_graph import SearchResult
from seshat.core.models.enums import SearchMode
from seshat.core.utils.hashing import fingerprint
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.core.config.settings import RAGConfig
    from seshat.infra.vector_store.base_store import AbstractVectorStore

logger = get_logger(__name__)


class SearchEngine:
    def __init__(
        self,
        rag_config: RAGConfig,
        vector_store: AbstractVectorStore,
        keyword_llm: BaseChatModel | None,
        multi_query_llm: BaseChatModel | None,
    ) -> None:
        self._rag_config = rag_config
        self._vs = vector_store
        self._keyword_agent = (
            KeywordAgent(keyword_llm, rag_config.keyword_extraction_llm)
            if keyword_llm is not None and rag_config.keyword_extraction_llm is not None
            else None
        )
        self._multi_query_agent = (
            MultiQueryAgent(multi_query_llm, rag_config.multi_query.llm, rag_config.multi_query.num_variants)
            if multi_query_llm is not None and rag_config.multi_query.llm is not None
            else None
        )

    @property
    def search_mode(self) -> SearchMode:
        return self._rag_config.search_mode

    async def search(
        self,
        query: str,
        *,
        node_filter: Any | None = None,
        exclude_job_id: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        logger.debug("search: mode=%s query=%r", self.search_mode.value, query[:60])
        common_search_kwargs = {
            "node_filter": node_filter,
            "exclude_job_id": exclude_job_id,
            "top_k": top_k if top_k is not None else self._rag_config.top_k,
        }

        match self.search_mode:
            case SearchMode.SEMANTIC:
                results = await self._semantic_search(query, score_threshold=score_threshold, **common_search_kwargs)
            case SearchMode.KEYWORD:
                results = await self._keyword_search(query, **common_search_kwargs)
            case SearchMode.HYBRID:
                results = await self._hybrid_search(query, score_threshold=score_threshold, **common_search_kwargs)
            case _:
                raise ValueError(f"Unsupported search mode: {self.search_mode.value!r}")

        logger.debug("search: returned %d results", len(results))
        return results

    async def _semantic_search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        variants = await self._generate_variants(query)
        if not variants:
            return await self._vs.search_dense(query, **kwargs)

        queries = [query, *variants]
        result_lists = await asyncio.gather(*[self._vs.search_dense(q, **kwargs) for q in queries])
        return _rrf(result_lists, [])

    async def _keyword_search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        keywords = await self._extract_keywords(query)
        return await self._vs.search_sparse(query=(keywords or query), **kwargs)

    async def _hybrid_search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        score_threshold = kwargs.pop("score_threshold", None)
        semantic, keyword = await asyncio.gather(
            self._semantic_search(query, score_threshold=score_threshold, **kwargs),
            self._keyword_search(query, **kwargs),
        )
        return _rrf([semantic], [keyword])

    async def _extract_keywords(self, query: str) -> str | None:
        if self._keyword_agent is not None:
            try:
                return await self._keyword_agent.extract(query)
            except Exception:
                logger.warning("keyword extraction LLM call failed; using original query")
                return None
        return query

    async def _generate_variants(self, query: str) -> list[str]:
        if self._multi_query_agent is not None:
            try:
                return await self._multi_query_agent.generate(query)
            except Exception:
                logger.warning("multi-query generation LLM call failed; using original query only")
                return []
        return []

    def fingerprint(self) -> str:
        """Stable hash over the retrieval config; used by eval to bust the result cache on any change."""
        parts = [
            self._rag_config.search_mode,
            self._keyword_agent.fingerprint() if self._keyword_agent else "none",
            self._multi_query_agent.fingerprint() if self._multi_query_agent else "none",
            (
                f"{self._rag_config.reranker.provider}:{self._rag_config.reranker.model}"
                if self._rag_config.reranker
                else "none"
            ),
        ]
        return fingerprint(":".join(parts))

    def prompt_texts(self) -> dict[str, str]:
        """Returns the active prompt strings keyed by role; used by MLflow to log the prompts alongside run params."""
        texts: dict[str, str] = {}
        if self._keyword_agent is not None:
            texts["keyword_extraction"] = self._keyword_agent.prompt_texts()["system"]
        if self._multi_query_agent is not None:
            texts["multi_query"] = self._multi_query_agent.prompt_texts()["system"]
        return texts


def _rrf(
    dense_leg: list[list[SearchResult]],
    sparse_leg: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion across multiple result lists. Score = sum(1 / (k + rank)) per node across all legs."""
    scores: dict[str, float] = {}
    for dense_results in dense_leg:
        for rank, result in enumerate(dense_results):
            node_id = str(result.node_id)
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)

    for sparse_results in sparse_leg:
        for rank, result in enumerate(sparse_results):
            node_id = str(result.node_id)
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [SearchResult(node_id=UUID(node_id), score=score) for node_id, score in ranked]
