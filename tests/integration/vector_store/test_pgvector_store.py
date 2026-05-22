import pytest

from seshat.models.api import NodeFilter
from seshat.models.enums import ConceptType
from seshat.vector_store.pgvector_store import PGVectorStore
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API, SKIP_IF_NO_POSTGRES

pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.embedding,
    SKIP_IF_NO_POSTGRES,
    SKIP_IF_NO_EMBEDDINGS_API,
]

_TEST_NODE_ID = "test-node-1"


class TestPGVectorStoreSearch:
    async def test_upsert_then_search(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await vector_store.search("PostgreSQL session storage", top_k=5)
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_with_node_type_filter_matching(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await vector_store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.DECISION),
        )
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_with_node_type_filter_nonmatching(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await vector_store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.RISK),
        )
        assert not any(r.node_id == _TEST_NODE_ID for r in results)


class TestPGVectorStoreDelete:
    async def test_delete(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use Redis for caching",
            {"node_type": "decision", "confidence": 0.8},
        )
        await vector_store.delete(_TEST_NODE_ID)
        results = await vector_store.search("Redis caching", top_k=5)
        assert not any(r.node_id == _TEST_NODE_ID for r in results)
