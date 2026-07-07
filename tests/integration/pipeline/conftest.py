import pytest

from seshat.core.config.settings import KBStoreConfig
from seshat.infra.knowledge_store.pg_store import PostgresKBStore
from seshat.infra.repositories.node_repository import NodeRepository


@pytest.fixture
async def kb_store(pg_test_url):
    store = PostgresKBStore(KBStoreConfig(), pg_test_url)
    await store.connect()
    yield store
    await store.pool.execute(f"TRUNCATE {store._schema}.kb_relationships, {store._schema}.kb_nodes CASCADE")
    await store.close()


@pytest.fixture
def node_repo(kb_store, vector_store) -> NodeRepository:
    return NodeRepository(kb_store, vector_store)
