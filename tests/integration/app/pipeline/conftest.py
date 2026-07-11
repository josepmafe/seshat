import pytest

from seshat.app.repositories.node_repository import NodeRepository
from seshat.core.config.settings import KBStoreConfig
from seshat.infra.knowledge_store.pg_store import PostgresKBStore


@pytest.fixture(scope="module")
async def kb_store(pg_test_url):
    store = PostgresKBStore(KBStoreConfig(), pg_test_url)
    await store.connect()
    yield store
    await store.close()


@pytest.fixture(autouse=True)
async def _truncate_kb_tables(kb_store):
    yield
    await kb_store.pool.execute(f"TRUNCATE {kb_store._schema}.kb_relationships, {kb_store._schema}.kb_nodes CASCADE")


@pytest.fixture
def node_repo(kb_store, vector_store) -> NodeRepository:
    return NodeRepository(kb_store, vector_store)
