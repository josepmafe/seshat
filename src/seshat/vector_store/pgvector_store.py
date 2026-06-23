from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import sqlalchemy
from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import Float, cast, func, select
from sqlalchemy.dialects.postgresql import TSVECTOR

from seshat.models.api import SearchResult
from seshat.models.enums import SearchMode
from seshat.utils.db import ensure_psycopg_scheme
from seshat.utils.log import get_logger
from seshat.vector_store.base_store import AbstractVectorStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from sqlalchemy.ext.asyncio import AsyncEngine

    from seshat.config.settings import VectorIndexConfig, VectorStoreConfig
    from seshat.models.api import NodeFilter

logger = get_logger(__name__)

# ts_content is our generated column (migration 004) — not declared on the ORM model,
# so we reference it as a bare column expression.
_TS_CONTENT = sqlalchemy.column("ts_content", TSVECTOR)


class KeywordSearchMixin:
    """FTS keyword search over langchain_pg_embedding via SQLAlchemy.

    Assumes the concrete class exposes:
      - self._engine: AsyncEngine
      - self._collection_name: str
      - self._embedding_store: the EmbeddingStore ORM class from PGVector
      - self._collection_store: the CollectionStore ORM class from PGVector
    and initializes self._collection_id to None.
    """

    _engine: AsyncEngine
    _collection_name: str
    _collection_id: str | None
    _embedding_store: Any
    _collection_store: Any

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None,
        exclude_job_id: str | None,
    ) -> list[SearchResult]:
        sparse = await self._sparse_search(query, top_k=top_k, node_filter=node_filter, exclude_job_id=exclude_job_id)
        return [SearchResult(node_id=nid, score=score) for nid, score in sparse]

    async def _sparse_search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None,
        exclude_job_id: str | None,
    ) -> list[tuple[str, float]]:
        ts_query_str = " | ".join(re.findall(r"\w+", query))
        if not ts_query_str:
            return []

        collection_id = await self._get_collection_id()
        ts_query_expr = func.to_tsquery("english", ts_query_str)
        ts_rank = func.ts_rank_cd(_TS_CONTENT, ts_query_expr)

        stmt = (
            select(
                self._embedding_store.cmetadata["node_id"].as_string().label("node_id"),
                ts_rank.label("rank"),
            )
            .where(
                self._embedding_store.collection_id == collection_id,
                _TS_CONTENT.op("@@")(ts_query_expr),
            )
            .order_by(ts_rank.desc())
            .limit(top_k)
        )
        stmt = self._apply_sparse_filter(stmt, node_filter, exclude_job_id)

        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()

        return [(row.node_id, float(row.rank)) for row in rows]

    async def _get_collection_id(self) -> str:
        if self._collection_id is None:
            CS = self._collection_store
            stmt = select(CS.uuid).where(CS.name == self._collection_name)
            async with self._engine.connect() as conn:
                row = (await conn.execute(stmt)).fetchone()
            if row is None:
                raise RuntimeError(f"Collection '{self._collection_name}' not found in langchain_pg_collection")
            self._collection_id = str(row.uuid)

        return self._collection_id

    def _apply_sparse_filter(self, stmt: Any, node_filter: NodeFilter | None, exclude_job_id: str | None) -> Any:
        self._embedding_store = self._embedding_store
        if node_filter is not None:
            if node_filter.node_type:
                stmt = stmt.where(self._embedding_store.cmetadata["node_type"].as_string() == node_filter.node_type)
            if node_filter.min_confidence is not None:
                stmt = stmt.where(
                    cast(self._embedding_store.cmetadata["confidence"].as_string(), Float) >= node_filter.min_confidence
                )
            if node_filter.ingestion_source:
                stmt = stmt.where(
                    self._embedding_store.cmetadata["ingestion_source"].as_string() == node_filter.ingestion_source
                )
        if exclude_job_id is not None:
            stmt = stmt.where(self._embedding_store.cmetadata["job_id"].as_string() != exclude_job_id)

        return stmt


class PGVectorStore(AbstractVectorStore, KeywordSearchMixin):
    def __init__(
        self, config: VectorStoreConfig, index: VectorIndexConfig, embeddings: Embeddings, connection_string: str
    ) -> None:
        self._config = config
        self._index = index
        self._connection_string = self._validate_connection_string(connection_string)
        self._store = PGVector(
            embeddings=embeddings, collection_name=index.collection, connection=self._connection_string, async_mode=True
        )
        self._collection_id: str | None = None

    @property
    def _engine(self) -> AsyncEngine:
        return self._store._async_engine

    @property
    def _collection_name(self) -> str:
        return self._index.collection

    @property
    def _embedding_store(self) -> Any:
        return self._store.EmbeddingStore

    @property
    def _collection_store(self) -> Any:
        return self._store.CollectionStore

    @staticmethod
    def _validate_connection_string(connection_string: str) -> str:
        return ensure_psycopg_scheme(
            connection_string,
            warn_msg=(
                "Unexpected driver %r in vector store connection string; "
                "replacing with '+psycopg' for langchain-postgres compatibility."
            ),
        )

    @staticmethod
    def get_supported_filter_fields() -> frozenset[str]:
        return frozenset({"node_type", "min_confidence", "ingestion_source"})

    async def upsert(self, node_id: str, text: str, metadata: dict) -> None:
        # TODO: assert metadata keys are a subset of get_supported_filter_fields()
        # TODO: delete embedding when node is archived/rejected (lifecycle sync)
        doc = Document(page_content=text, metadata={**metadata, "node_id": node_id})
        await self._store.aadd_documents([doc], ids=[node_id])

    async def search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
        score_threshold: float | None = None,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> list[SearchResult]:
        match mode:
            case SearchMode.KEYWORD:
                return await self._keyword_search(query, top_k, node_filter, exclude_job_id)
            case SearchMode.SEMANTIC:
                return await self._semantic_search(query, top_k, node_filter, exclude_job_id, score_threshold)
            case SearchMode.HYBRID:
                return await self._hybrid_search(query, top_k, node_filter, exclude_job_id, score_threshold)

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None,
        exclude_job_id: str | None,
        score_threshold: float | None,
    ) -> list[SearchResult]:
        results = await self._similarity_search(
            query, top_k=top_k, node_filter=node_filter, exclude_job_id=exclude_job_id, score_threshold=score_threshold
        )
        return [SearchResult(node_id=doc.metadata["node_id"], score=score) for doc, score in results]

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None,
        exclude_job_id: str | None,
        score_threshold: float | None,
    ) -> list[SearchResult]:
        search_kwargs = {"top_k": top_k, "node_filter": node_filter, "exclude_job_id": exclude_job_id}
        dense = await self._similarity_search(query, score_threshold=score_threshold, **search_kwargs)
        sparse = await self._sparse_search(query, **search_kwargs)
        return _rrf(dense, sparse, top_k=top_k)

    async def _similarity_search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None,
        exclude_job_id: str | None,
        score_threshold: float | None,
    ) -> list[tuple[Document, float]]:
        return await self._store.asimilarity_search_with_relevance_scores(
            query, k=top_k, filter=_build_semantic_filter(node_filter, exclude_job_id), score_threshold=score_threshold
        )

    async def delete(self, node_id: str) -> None:
        await self._store.adelete(ids=[node_id])


def _rrf(
    dense: list[tuple[Document, float]],
    sparse: list[tuple[str, float]],
    top_k: int,
    k: int = 60,
) -> list[SearchResult]:
    """Merge dense and sparse ranked lists via Reciprocal Rank Fusion.

    Each result scores 1/(k + rank) per list it appears in; scores are summed, so a
    node ranked highly in both legs outscores one ranked highly in only one. k=60 is
    the empirically validated default from the original paper.

    Reference: Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods." SIGIR 2009.
    https://dl.acm.org/doi/10.1145/1571941.1572114
    """
    scores: dict[str, float] = {}
    for rank, (doc, _) in enumerate(dense):
        nid = doc.metadata["node_id"]
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)

    for rank, (nid, _) in enumerate(sparse):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [SearchResult(node_id=nid, score=score) for nid, score in ranked[:top_k]]


def _build_semantic_filter(node_filter: NodeFilter | None, exclude_job_id: str | None = None) -> dict | None:
    if node_filter is None and exclude_job_id is None:
        return None

    result: dict[str, Any] = {}

    if node_filter is not None:
        unsupported = {f for f in node_filter.model_fields_set if f not in PGVectorStore.get_supported_filter_fields()}
        if unsupported:
            raise NotImplementedError(
                f"PGVector metadata filter does not support: {sorted(unsupported)}. "
                "Use PostgresKBStore.query() for full NodeFilter support."
            )
        if node_filter.node_type:
            result["node_type"] = node_filter.node_type.value
        if node_filter.min_confidence is not None:
            result["confidence"] = {"$gte": node_filter.min_confidence}
        if node_filter.ingestion_source:
            result["ingestion_source"] = node_filter.ingestion_source.value

    if exclude_job_id is not None:
        result["job_id"] = {"$ne": exclude_job_id}

    return result
