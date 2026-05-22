import pytest

from seshat.config.settings import KBStoreConfig
from seshat.knowledge_store.pg_store import PostgresKBStore


@pytest.fixture
async def kb_store(pg_test_url):
    store = PostgresKBStore(KBStoreConfig(), pg_test_url)
    await store.connect()
    yield store
    await store.pool.execute(f"TRUNCATE {store._schema}.kb_relationships, {store._schema}.kb_nodes CASCADE")
    await store.close()
