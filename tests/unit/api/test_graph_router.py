from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from seshat.api.app import create_app
from seshat.api.dependencies import _get_current_user, get_app_state
from seshat.api.state import AppState
from seshat.models.enums import NodeState, UserRole
from tests.helpers import make_node


def _make_app_state() -> AppState:
    kb_store = MagicMock()
    kb_store.query = AsyncMock(return_value=[])
    kb_store.get_node = AsyncMock(return_value=None)
    kb_store.get_neighbours = AsyncMock(return_value=[])

    return AppState(
        ops=MagicMock(),
        kb_store=kb_store,
        config=MagicMock(),
        queue=MagicMock(),
        results={},
        runner=MagicMock(),
    )


def _make_current_user(role: UserRole = UserRole.OPERATOR):
    from seshat.api.dependencies import CurrentUser

    return CurrentUser(user_id="alice", role=role)


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test/v1")


def _override(app, state: AppState, user=None):
    app.dependency_overrides[get_app_state] = lambda: state
    if user is not None:
        app.dependency_overrides[_get_current_user] = lambda: user


def _clear(app):
    app.dependency_overrides.clear()


class TestQueryGraph:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.get("/graph")
        _clear(app)
        assert resp.status_code == 401

    async def test_returns_empty_nodes(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/graph")
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["nodes"] == []

    async def test_returns_matching_nodes(self, app, client):
        node = make_node()
        state = _make_app_state()
        state.kb_store.query = AsyncMock(return_value=[node])
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/graph")
        _clear(app)
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) == 1


class TestGetNode:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.get("/graph/some-node-id")
        _clear(app)
        assert resp.status_code == 401

    async def test_not_found(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/graph/nonexistent")
        _clear(app)
        assert resp.status_code == 404

    async def test_returns_node_with_neighbours(self, app, client):
        node = make_node()
        neighbour = make_node("n2")
        state = _make_app_state()
        state.kb_store.get_node = AsyncMock(return_value=node)
        state.kb_store.get_neighbours = AsyncMock(return_value=[neighbour])
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get(f"/graph/{node.id}")
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["node"]["id"] == str(node.id)
        assert len(resp.json()["neighbours"]) == 1

    async def test_filters_non_current_neighbours(self, app, client):
        from tests.helpers import make_node

        node = make_node()
        superseded = make_node("n2")
        superseded_meta = superseded.metadata.model_copy(update={})
        superseded = superseded.model_copy(update={"state": NodeState.SUPERSEDED, "metadata": superseded_meta})

        state = _make_app_state()
        state.kb_store.get_node = AsyncMock(return_value=node)
        state.kb_store.get_neighbours = AsyncMock(return_value=[superseded])
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get(f"/graph/{node.id}")
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["neighbours"] == []


class TestImpactTraversal:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.get("/graph/some-node-id/impact")
        _clear(app)
        assert resp.status_code == 401

    async def test_returns_empty_when_no_neighbours(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/graph/some-node-id/impact")
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["nodes"] == []

    async def test_traverses_inbound_neighbours(self, app, client):
        node = make_node()
        neighbour = make_node("n2", confidence=0.9)
        state = _make_app_state()
        state.kb_store.get_neighbours = AsyncMock(return_value=[neighbour])
        state.kb_store.get_node = AsyncMock(return_value=neighbour)
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get(f"/graph/{node.id}/impact?depth=1")
        _clear(app)
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) == 1
        assert resp.json()["nodes"][0]["traversal_depth"] == 1

    async def test_depth_out_of_range(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/graph/some-node-id/impact?depth=10")
        _clear(app)
        assert resp.status_code == 422
