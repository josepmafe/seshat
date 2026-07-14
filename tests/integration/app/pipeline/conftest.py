from uuid import uuid4

import pytest
import pytest_asyncio

from seshat.app.repositories.node_repository import NodeRepository
from seshat.core.config.settings import KBStoreConfig, SecretsConfig, SeshatConfig, VectorStoreConfig
from seshat.core.models.enums import SecretsProvider
from seshat.infra.knowledge_store.pg_store import PostgresKBStore
from seshat.infra.vector_store.factory import _build_embeddings
from seshat.infra.vector_store.pgvector_store import PGVectorStore


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def kb_store(pg_test_url):
    store = PostgresKBStore(KBStoreConfig(), pg_test_url)
    await store.connect()
    yield store
    await store.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def vector_store(pg_test_url):
    seshat_config = SeshatConfig(secrets=SecretsConfig(provider=SecretsProvider.ENV))
    index = seshat_config.vector_index.model_copy(update={"collection": f"test_pipeline_{uuid4().hex}"})
    embeddings = _build_embeddings(index, seshat_config)
    store = PGVectorStore(VectorStoreConfig(), index, embeddings, pg_test_url)
    yield store
    await store._store.adelete_collection()


@pytest_asyncio.fixture(autouse=True, loop_scope="module")
async def _truncate_kb_tables(kb_store):
    yield
    await kb_store.pool.execute(f"TRUNCATE {kb_store._schema}.kb_relationships, {kb_store._schema}.kb_nodes CASCADE")


@pytest_asyncio.fixture(autouse=True, loop_scope="module")
async def _reset_vector_store(vector_store):
    yield
    await vector_store._store.adelete_collection()
    await vector_store._store.acreate_collection()


@pytest.fixture(scope="module")
def node_repo(kb_store, vector_store) -> NodeRepository:
    return NodeRepository(kb_store, vector_store)
