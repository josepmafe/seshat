from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from seshat.app.platform.api.app import create_app
from seshat.app.platform.api.dependencies import CurrentUser, _get_current_user, get_app_state
from seshat.app.platform.api.state import AppState
from seshat.app.repositories.node_repository import NodeRepository
from seshat.app.services.graph import GraphService
from seshat.core.config.settings import KBStoreConfig
from seshat.core.models.enums import UserRole
from seshat.infra.knowledge_store.pg_store import PostgresKBStore
from tests.integration.conftest import SKIP_IF_NO_POSTGRES

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]

_CREATE_PAYLOAD = {"type": "decision", "title": "Use PostgreSQL", "description": "Chosen for ACID compliance"}
_OPERATOR = CurrentUser(user_id="alice", role=UserRole.OPERATOR)
_ADMIN = CurrentUser(user_id="admin", role=UserRole.ADMIN)


@pytest.fixture
async def kb_store(pg_test_url):
    config = KBStoreConfig()
    store = PostgresKBStore(config, pg_test_url)
    await store.connect()
    yield store
    await store.pool.execute(f"TRUNCATE {store._schema}.kb_relationships, {store._schema}.kb_nodes CASCADE")
    await store.close()


@pytest.fixture
def fake_vector_store():
    vs = MagicMock()
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()
    vs.search = AsyncMock(return_value=[])
    return vs


@pytest.fixture
def fake_extraction_orch():
    orch = MagicMock()
    orch.run_resolution = AsyncMock()
    return orch


@pytest.fixture
def graph_svc(kb_store, fake_vector_store, fake_extraction_orch):
    node_repo = NodeRepository(kb_store, fake_vector_store)
    return GraphService(node_repo, fake_extraction_orch)


@pytest.fixture
def app_state(graph_svc):
    return AppState(
        config=MagicMock(),
        admin_service=MagicMock(),
        health_service=MagicMock(),
        graph_service=graph_svc,
        job_service=MagicMock(),
    )


@pytest.fixture
def fastapi_app():
    return create_app()


def _client(fastapi_app, state: AppState, user: CurrentUser) -> AsyncClient:
    fastapi_app.dependency_overrides[get_app_state] = lambda: state
    fastapi_app.dependency_overrides[_get_current_user] = lambda: user
    return AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test/v1")


class TestGraphCRUDRoundTrip:
    async def test_create_then_get_then_list(self, fastapi_app, app_state):
        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            create_resp = await ac.post("/graph/nodes", json=_CREATE_PAYLOAD)

        assert create_resp.status_code == 201
        node_id = create_resp.json()["id"]

        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            get_resp = await ac.get(f"/graph/{node_id}")

        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == node_id
        assert get_resp.json()["title"] == "Use PostgreSQL"

        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            list_resp = await ac.get("/graph")

        assert list_resp.status_code == 200
        listed_ids = [n["id"] for n in list_resp.json()["nodes"]]
        assert node_id in listed_ids

        fastapi_app.dependency_overrides.clear()

    async def test_nonexistent_id_returns_404(self, fastapi_app, app_state):
        missing_id = "00000000-0000-0000-0000-000000000000"

        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            resp = await ac.get(f"/graph/{missing_id}")

        assert resp.status_code == 404

        fastapi_app.dependency_overrides.clear()

    async def test_status_filter_exercises_sql_enum_coercion(self, fastapi_app, app_state):
        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            create_resp = await ac.post("/graph/nodes", json=_CREATE_PAYLOAD)

        node_id = create_resp.json()["id"]

        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            approved_resp = await ac.get("/graph?status=approved")

        assert approved_resp.status_code == 200
        listed_ids = [n["id"] for n in approved_resp.json()["nodes"]]
        assert node_id in listed_ids

        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            superseded_resp = await ac.get("/graph?state=superseded")

        assert superseded_resp.status_code == 200
        listed_superseded_ids = [n["id"] for n in superseded_resp.json()["nodes"]]
        assert node_id not in listed_superseded_ids

        fastapi_app.dependency_overrides.clear()

    async def test_delete_then_get_returns_404(self, fastapi_app, app_state):
        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            create_resp = await ac.post("/graph/nodes", json=_CREATE_PAYLOAD)

        node_id = create_resp.json()["id"]

        async with _client(fastapi_app, app_state, _ADMIN) as ac:
            delete_resp = await ac.delete(f"/graph/nodes/{node_id}")

        assert delete_resp.status_code == 204

        async with _client(fastapi_app, app_state, _OPERATOR) as ac:
            get_resp = await ac.get(f"/graph/{node_id}")

        assert get_resp.status_code == 404

        fastapi_app.dependency_overrides.clear()
