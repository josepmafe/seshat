from __future__ import annotations

import logging

import pytest
from langchain_core.documents import Document

from seshat.vector_store.pgvector_store import PGVectorStore, _rrf


class TestRrf:
    def test_node_in_both_legs_scores_higher_than_node_in_one_leg(self):
        dense = [
            (Document(page_content="x", metadata={"node_id": "n1"}), 0.9),
            (Document(page_content="x", metadata={"node_id": "n2"}), 0.8),
        ]
        sparse = [("n1", 0.5), ("n3", 0.4)]
        results = _rrf(dense, sparse, top_k=3)
        assert results[0].node_id == "n1"

    def test_rrf_truncates_to_top_k(self):
        dense = [(Document(page_content="x", metadata={"node_id": f"n{i}"}), 0.9 - i * 0.1) for i in range(4)]
        results = _rrf(dense, [], top_k=2)
        assert len(results) == 2

    def test_rrf_node_only_in_sparse_is_included(self):
        dense = [(Document(page_content="x", metadata={"node_id": "n1"}), 0.9)]
        sparse = [("n2", 0.7)]
        results = _rrf(dense, sparse, top_k=2)
        assert any(r.node_id == "n2" for r in results)

    def test_rrf_empty_inputs_returns_empty(self):
        assert _rrf([], [], top_k=5) == []

    def test_rrf_scores_are_positive_floats(self):
        dense = [(Document(page_content="x", metadata={"node_id": "n1"}), 0.9)]
        sparse = [("n1", 0.5)]
        results = _rrf(dense, sparse, top_k=1)
        assert results[0].score > 0


class TestSparseSearchGuard:
    @pytest.mark.asyncio
    async def test_no_extractor_logs_warning_and_returns_empty(self, caplog):
        store = PGVectorStore.__new__(PGVectorStore)
        store._keyword_extractor = None

        with caplog.at_level(logging.WARNING, logger="seshat.vector_store.pgvector_store"):
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


class TestSearchModeRouting:
    def test_score_threshold_not_in_rrf_signature(self):
        import inspect

        sig = inspect.signature(_rrf)
        assert "score_threshold" not in sig.parameters


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
        with caplog.at_level(logging.WARNING, logger="seshat.vector_store.pgvector_store"):
            PGVectorStore._validate_connection_string("postgresql+asyncpg://user:pass@host/db")
        assert "+asyncpg" in caplog.text

    def test_plain_scheme_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="seshat.vector_store.pgvector_store"):
            PGVectorStore._validate_connection_string("postgresql://user:pass@host/db")
        assert caplog.text == ""

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid connection string"):
            PGVectorStore._validate_connection_string("mysql://user:pass@host/db")

    def test_error_message_does_not_contain_credentials(self):
        with pytest.raises(ValueError, match="Invalid connection string") as exc_info:
            PGVectorStore._validate_connection_string("mysql://secret:hunter2@host/db")
        assert "secret" not in str(exc_info.value)
        assert "hunter2" not in str(exc_info.value)
