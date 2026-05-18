import pytest

from seshat.config.settings import SecretsConfig, SeshatConfig, VectorStoreConfig
from seshat.models.api import NodeFilter
from seshat.models.enums import ConceptType, SecretsProvider
from seshat.vector_store.factory import _build_embeddings
from seshat.vector_store.pgvector_store import PGVectorStore
from tests.integration.conftest import SKIP_IF_NO_LLM_API, SKIP_IF_NO_POSTGRES

pytestmark = [pytest.mark.integration, pytest.mark.llm, SKIP_IF_NO_POSTGRES, SKIP_IF_NO_LLM_API]

_TEST_NODE_ID = "test-node-1"


@pytest.fixture
async def store(pg_test_url):
    seshat_config = SeshatConfig(secrets=SecretsConfig(provider=SecretsProvider.ENV))
    index = seshat_config.vector_index.model_copy(update={"collection": "test_collection"})
    embeddings = _build_embeddings(index, seshat_config)
    s = PGVectorStore(VectorStoreConfig(), index, embeddings, pg_test_url)
    yield s
    await s.delete(_TEST_NODE_ID)


class TestPGVectorStoreSearch:
    async def test_upsert_then_search(self, store: PGVectorStore):
        await store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await store.search("PostgreSQL session storage", top_k=5)
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_with_node_type_filter_matching(self, store: PGVectorStore):
        await store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.DECISION),
        )
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_with_node_type_filter_nonmatching(self, store: PGVectorStore):
        await store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.RISK),
        )
        assert not any(r.node_id == _TEST_NODE_ID for r in results)


class TestPGVectorStoreDelete:
    async def test_delete(self, store: PGVectorStore):
        await store.upsert(
            _TEST_NODE_ID,
            "Use Redis for caching",
            {"node_type": "decision", "confidence": 0.8},
        )
        await store.delete(_TEST_NODE_ID)
        results = await store.search("Redis caching", top_k=5)
        assert not any(r.node_id == _TEST_NODE_ID for r in results)
