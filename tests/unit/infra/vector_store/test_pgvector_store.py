from __future__ import annotations

import logging
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from langchain_core.documents import Document

from seshat.core.models.api_graph import NodeFilter
from seshat.infra.vector_store.pgvector_store import PGVectorStore, _rrf
from tests.unit.infra.helpers import assert_credentials_not_in_error, assert_invalid_scheme_raises

_N1 = "00000000-0000-0000-0000-000000000001"
_N2 = "00000000-0000-0000-0000-000000000002"
_N3 = "00000000-0000-0000-0000-000000000003"


class TestRrf:
    def test_node_in_both_legs_scores_higher_than_node_in_one_leg(self):
        dense = [
            (Document(page_content="x", metadata={"node_id": _N1}), 0.9),
            (Document(page_content="x", metadata={"node_id": _N2}), 0.8),
        ]
        sparse = [(_N1, 0.5), (_N3, 0.4)]
        results = _rrf(dense, sparse, top_k=3)
        assert results[0].node_id == UUID(_N1)

    def test_rrf_truncates_to_top_k(self):
        uids = [f"00000000-0000-0000-0000-00000000000{i}" for i in range(4)]
        dense = [(Document(page_content="x", metadata={"node_id": uid}), 0.9 - i * 0.1) for i, uid in enumerate(uids)]
        results = _rrf(dense, [], top_k=2)
        assert len(results) == 2

    def test_rrf_node_only_in_sparse_is_included(self):
        dense = [(Document(page_content="x", metadata={"node_id": _N1}), 0.9)]
        sparse = [(_N2, 0.7)]
        results = _rrf(dense, sparse, top_k=2)
        assert any(r.node_id == UUID(_N2) for r in results)

    def test_rrf_empty_inputs_returns_empty(self):
        assert _rrf([], [], top_k=5) == []

    def test_rrf_scores_are_positive_floats(self):
        dense = [(Document(page_content="x", metadata={"node_id": _N1}), 0.9)]
        sparse = [(_N1, 0.5)]
        results = _rrf(dense, sparse, top_k=1)
        assert results[0].score > 0


class TestSparseSearchGuard:
    @pytest.mark.asyncio
    async def test_no_extractor_logs_warning_and_returns_empty(self, caplog):
        store = PGVectorStore.__new__(PGVectorStore)
        store._keyword_extractor = None

        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            result = await store._sparse_search("some query", top_k=5, node_filter=None, exclude_job_id=None)

        assert result == []
        assert "keyword_extractor" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_without_hitting_extractor(self):
        called = []

        async def extractor(q):
            called.append(q)
            return "keywords"

        store = PGVectorStore.__new__(PGVectorStore)
        store._keyword_extractor = extractor

        result = await store._sparse_search("   ", top_k=5, node_filter=None, exclude_job_id=None)

        assert result == []
        assert called == []

    @pytest.mark.asyncio
    async def test_missing_collection_propagates_from_sparse_search(self):
        store = PGVectorStore.__new__(PGVectorStore)
        store._keyword_extractor = AsyncMock(return_value="budget approval")
        store._ts_content_ready = True
        store._collection_id = None
        store._get_collection_id = AsyncMock(
            side_effect=RuntimeError("Collection 'seshat_kb' not found in langchain_pg_collection")
        )
        store._ensure_ts_content = AsyncMock()

        with pytest.raises(RuntimeError, match="seshat_kb"):
            await store._sparse_search("budget approval", top_k=5, node_filter=None, exclude_job_id=None)


class TestValidateConnectionString:
    def test_psycopg_qualifier_accepted_unchanged(self):
        result = PGVectorStore._validate_connection_string("postgresql+psycopg://user:pass@host/db")
        assert result == "postgresql+psycopg://user:pass@host/db"

    def test_plain_scheme_gets_psycopg_qualifier(self):
        result = PGVectorStore._validate_connection_string("postgresql://user:pass@host/db")
        assert result == "postgresql+psycopg://user:pass@host/db"

    def test_wrong_qualifier_replaced(self):
        result = PGVectorStore._validate_connection_string("postgresql+asyncpg://user:pass@host/db")
        assert result == "postgresql+psycopg://user:pass@host/db"

    def test_wrong_qualifier_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            PGVectorStore._validate_connection_string("postgresql+asyncpg://user:pass@host/db")
        assert "+asyncpg" in caplog.text

    def test_plain_scheme_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            PGVectorStore._validate_connection_string("postgresql://user:pass@host/db")
        assert caplog.text == ""

    def test_invalid_scheme_raises(self):
        assert_invalid_scheme_raises(PGVectorStore)

    def test_error_message_does_not_contain_credentials(self):
        assert_credentials_not_in_error(PGVectorStore)

    def test_psycopg2_qualifier_replaced(self):
        result = PGVectorStore._validate_connection_string("postgresql+psycopg2://user:pass@host/db")
        assert result == "postgresql+psycopg://user:pass@host/db"


class TestBuildSemanticFilter:
    def _store(self) -> PGVectorStore:
        return PGVectorStore.__new__(PGVectorStore)

    def test_none_filter_and_no_exclude_returns_none(self):
        assert self._store()._build_semantic_filter(None) is None

    def test_supported_node_type_filter_applied(self):
        from seshat.core.models.enums import ConceptType

        nf = NodeFilter(node_type=ConceptType.DECISION)
        result = self._store()._build_semantic_filter(nf)
        assert result == {"node_type": ConceptType.DECISION.value}

    def test_supported_min_confidence_filter_applied(self):
        nf = NodeFilter(min_confidence=0.7)
        result = self._store()._build_semantic_filter(nf)
        assert result == {"confidence": {"$gte": 0.7}}

    def test_unsupported_fields_warn_and_are_ignored(self, caplog):
        from seshat.core.models.enums import NodeStatus

        nf = NodeFilter(status=NodeStatus.APPROVED)
        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            result = self._store()._build_semantic_filter(nf)

        assert "status" in caplog.text
        assert "supported" in caplog.text
        assert result == {}

    def test_unsupported_fields_do_not_prevent_supported_fields_from_applying(self, caplog):
        from seshat.core.models.enums import ConceptType, NodeStatus

        nf = NodeFilter(node_type=ConceptType.DECISION, status=NodeStatus.APPROVED)
        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            result = self._store()._build_semantic_filter(nf)

        assert result == {"node_type": ConceptType.DECISION.value}
        assert "status" in caplog.text

    def test_exclude_job_id_adds_ne_filter(self):
        result = self._store()._build_semantic_filter(None, exclude_job_id="job-123")
        assert result == {"job_id": {"$ne": "job-123"}}

    def test_no_active_filters_returns_empty_dict_not_none(self):
        # node_filter set but all supported fields are None → returns {} (not None)
        nf = NodeFilter()
        result = self._store()._build_semantic_filter(nf)
        assert result == {}


class TestRrfDuplicateHandling:
    def test_duplicate_node_id_in_dense_does_not_double_count(self):
        """The same node_id appearing twice in dense results should only score once."""
        doc = Document(page_content="x", metadata={"node_id": _N1})
        dense = [(doc, 0.9), (doc, 0.8)]
        results = _rrf(dense, [], top_k=5)
        matching = [r for r in results if r.node_id == UUID(_N1)]
        assert len(matching) == 1
