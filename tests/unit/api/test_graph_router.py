from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from seshat.api.app import create_app
from seshat.api.dependencies import CurrentUser, _get_current_user, get_app_state
from seshat.api.state import AppState
from seshat.models.api import BulkFailure, BulkResult
from seshat.models.enums import NodeState, UserRole
from seshat.worker.manual_ingestion import NodeNotFoundError, NodePreconditionError
from tests.helpers import make_node


def _make_app_state() -> AppState:
    kb_store = MagicMock()
    kb_store.query = AsyncMock(return_value=[])
    kb_store.get_node = AsyncMock(return_value=None)
    kb_store.get_neighbours = AsyncMock(return_value=[])

    manual_ingestion = MagicMock()
    manual_ingestion.create = AsyncMock()
    manual_ingestion.update = AsyncMock()
    manual_ingestion.override = AsyncMock()
    manual_ingestion.delete = AsyncMock()
    manual_ingestion.bulk_create = AsyncMock()
    manual_ingestion.bulk_delete = AsyncMock()

    return AppState(
        ops=MagicMock(),
        kb_store=kb_store,
        config=MagicMock(),
        queue=MagicMock(),
        results={},
        runner=MagicMock(),
        manual_ingestion=manual_ingestion,
    )


def _make_current_user(role: UserRole = UserRole.OPERATOR):
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
        node = make_node()
        superseded = make_node("n2")
        superseded = superseded.model_copy(
            update={"state": NodeState.SUPERSEDED, "metadata": superseded.metadata.model_copy(update={})}
        )
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


class TestCreateNode:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.post("/graph", json={"type": "decision", "title": "T", "description": "D"})
        _clear(app)
        assert resp.status_code == 401

    async def test_viewer_cannot_create(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.VIEWER))
        async with client as ac:
            resp = await ac.post("/graph", json={"type": "decision", "title": "T", "description": "D"})
        _clear(app)
        assert resp.status_code == 403

    async def test_returns_201_with_node(self, app, client):
        node = make_node()
        state = _make_app_state()
        state.manual_ingestion.create = AsyncMock(return_value=node)
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post("/graph", json={"type": "decision", "title": "T", "description": "D"})
        _clear(app)
        assert resp.status_code == 201
        assert resp.json()["id"] == str(node.id)

    async def test_passes_user_id_to_service(self, app, client):
        node = make_node()
        state = _make_app_state()
        state.manual_ingestion.create = AsyncMock(return_value=node)
        _override(app, state, _make_current_user())
        async with client as ac:
            await ac.post("/graph", json={"type": "decision", "title": "T", "description": "D"})
        _clear(app)
        state.manual_ingestion.create.assert_called_once()
        assert state.manual_ingestion.create.call_args.args[1] == "alice"


class TestUpdateNode:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        _clear(app)
        assert resp.status_code == 401

    async def test_viewer_cannot_update(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.VIEWER))
        async with client as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        _clear(app)
        assert resp.status_code == 403

    async def test_returns_updated_node(self, app, client):
        node = make_node()
        state = _make_app_state()
        state.manual_ingestion.update = AsyncMock(return_value=node)
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["id"] == str(node.id)

    async def test_not_found_returns_404(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.update = AsyncMock(side_effect=NodeNotFoundError("node-1"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        _clear(app)
        assert resp.status_code == 404

    async def test_precondition_failure_returns_409(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.update = AsyncMock(side_effect=NodePreconditionError("not manual"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        _clear(app)
        assert resp.status_code == 409
        assert "not manual" in resp.json()["detail"]


class TestOverrideNode:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        _clear(app)
        assert resp.status_code == 401

    async def test_viewer_cannot_override(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.VIEWER))
        async with client as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        _clear(app)
        assert resp.status_code == 403

    async def test_returns_updated_node(self, app, client):
        node = make_node()
        state = _make_app_state()
        state.manual_ingestion.override = AsyncMock(return_value=node)
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["id"] == str(node.id)

    async def test_operator_gets_auto_minimum_method(self, app, client):
        from seshat.models.enums import ApprovalMethod

        node = make_node()
        state = _make_app_state()
        state.manual_ingestion.override = AsyncMock(return_value=node)
        _override(app, state, _make_current_user(role=UserRole.OPERATOR))
        async with client as ac:
            await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        _clear(app)
        assert state.manual_ingestion.override.call_args.kwargs["minimum_method"] == ApprovalMethod.AUTO

    async def test_admin_gets_none_minimum_method(self, app, client):
        node = make_node()
        state = _make_app_state()
        state.manual_ingestion.override = AsyncMock(return_value=node)
        _override(app, state, _make_current_user(role=UserRole.ADMIN))
        async with client as ac:
            await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        _clear(app)
        assert state.manual_ingestion.override.call_args.kwargs["minimum_method"] is None

    async def test_not_found_returns_404(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.override = AsyncMock(side_effect=NodeNotFoundError("node-1"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        _clear(app)
        assert resp.status_code == 404

    async def test_precondition_failure_returns_409(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.override = AsyncMock(side_effect=NodePreconditionError("insufficient role"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        _clear(app)
        assert resp.status_code == 409


class TestDeleteNode:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.delete("/graph/node-1")
        _clear(app)
        assert resp.status_code == 401

    async def test_operator_cannot_delete(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.OPERATOR))
        async with client as ac:
            resp = await ac.delete("/graph/node-1")
        _clear(app)
        assert resp.status_code == 403

    async def test_returns_204(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.ADMIN))
        async with client as ac:
            resp = await ac.delete("/graph/node-1")
        _clear(app)
        assert resp.status_code == 204

    async def test_cascade_true_by_default(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.ADMIN))
        async with client as ac:
            await ac.delete("/graph/node-1")
        _clear(app)
        state.manual_ingestion.delete.assert_called_once()
        assert state.manual_ingestion.delete.call_args.kwargs.get("cascade") is True

    async def test_cascade_false_when_specified(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.ADMIN))
        async with client as ac:
            await ac.delete("/graph/node-1?cascade=false")
        _clear(app)
        assert state.manual_ingestion.delete.call_args.kwargs.get("cascade") is False

    async def test_precondition_failure_returns_409(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.delete = AsyncMock(side_effect=NodePreconditionError("has inbound"))
        _override(app, state, _make_current_user(role=UserRole.ADMIN))
        async with client as ac:
            resp = await ac.delete("/graph/node-1?cascade=false")
        _clear(app)
        assert resp.status_code == 409
        assert "has inbound" in resp.json()["detail"]


class TestBulkCreateNodes:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.post(
                "/graph/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        _clear(app)
        assert resp.status_code == 401

    async def test_viewer_cannot_bulk_create(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.VIEWER))
        async with client as ac:
            resp = await ac.post(
                "/graph/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        _clear(app)
        assert resp.status_code == 403

    async def test_returns_bulk_result(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.bulk_create = AsyncMock(return_value=BulkResult(succeeded=["uuid-1"], failed=[]))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post(
                "/graph/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["succeeded"] == ["uuid-1"]
        assert resp.json()["failed"] == []

    async def test_passes_user_id_to_service(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.bulk_create = AsyncMock(return_value=BulkResult(succeeded=[], failed=[]))
        _override(app, state, _make_current_user())
        async with client as ac:
            await ac.post("/graph/bulk", json={"nodes": [], "on_error": "continue"})
        _clear(app)
        state.manual_ingestion.bulk_create.assert_called_once()
        assert state.manual_ingestion.bulk_create.call_args.args[1] == "alice"


class TestBulkDeleteNodes:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.request("DELETE", "/graph/bulk", json={"node_ids": ["id-1"]})
        _clear(app)
        assert resp.status_code == 401

    async def test_operator_cannot_bulk_delete(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.OPERATOR))
        async with client as ac:
            resp = await ac.request("DELETE", "/graph/bulk", json={"node_ids": ["id-1"]})
        _clear(app)
        assert resp.status_code == 403

    async def test_returns_bulk_result(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.bulk_delete = AsyncMock(return_value=BulkResult(succeeded=["id-1"], failed=[]))
        _override(app, state, _make_current_user(role=UserRole.ADMIN))
        async with client as ac:
            resp = await ac.request("DELETE", "/graph/bulk", json={"node_ids": ["id-1"]})
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["succeeded"] == ["id-1"]

    async def test_cascade_passed_to_service(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.bulk_delete = AsyncMock(return_value=BulkResult(succeeded=[], failed=[]))
        _override(app, state, _make_current_user(role=UserRole.ADMIN))
        async with client as ac:
            await ac.request("DELETE", "/graph/bulk?cascade=false", json={"node_ids": []})
        _clear(app)
        assert state.manual_ingestion.bulk_delete.call_args.kwargs.get("cascade") is False

    async def test_partial_failure_in_result(self, app, client):
        state = _make_app_state()
        state.manual_ingestion.bulk_delete = AsyncMock(
            return_value=BulkResult(
                succeeded=["id-1"],
                failed=[BulkFailure(node_id="id-2", error="not found")],
            )
        )
        _override(app, state, _make_current_user(role=UserRole.ADMIN))
        async with client as ac:
            resp = await ac.request("DELETE", "/graph/bulk", json={"node_ids": ["id-1", "id-2"]})
        _clear(app)
        assert resp.json()["failed"][0]["node_id"] == "id-2"
