from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from seshat.api.state import AppState
from seshat.models.api_graph import BulkFailure, BulkResult
from seshat.models.api_responses import ImpactNode, ImpactResponse, NodeDetailResponse
from seshat.models.enums import ApprovalMethod, RelationshipType, UserRole
from seshat.services.graph_service import NodeNotFoundError, NodePreconditionError
from tests.helpers import make_node
from tests.unit.api.conftest import make_current_user


def _make_app_state() -> AppState:
    graph_service = MagicMock()
    graph_service.query = AsyncMock(return_value=[])
    graph_service.get_node_detail = AsyncMock(side_effect=NodeNotFoundError("not found"))
    graph_service.traverse_impact = AsyncMock(return_value=ImpactResponse(nodes=[]))
    graph_service.create = AsyncMock()
    graph_service.update = AsyncMock()
    graph_service.override = AsyncMock()
    graph_service.delete = AsyncMock()
    graph_service.bulk_create = AsyncMock()
    graph_service.bulk_delete = AsyncMock()
    graph_service.resolve = AsyncMock(return_value=[])
    graph_service.resolve_by_ids = AsyncMock(return_value=0)

    return AppState(
        config=MagicMock(),
        admin_service=MagicMock(),
        health_service=MagicMock(),
        graph_service=graph_service,
        job_service=MagicMock(),
    )


class TestQueryGraph:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/graph")
        assert resp.status_code == 401

    async def test_returns_empty_nodes(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/graph")
        assert resp.status_code == 200
        assert resp.json()["nodes"] == []

    async def test_returns_matching_nodes(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.query = AsyncMock(return_value=[node])
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/graph")
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) == 1

    async def test_passes_status_filter(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/graph?status=approved")
        assert resp.status_code == 200
        called_filter = state.graph_service.query.call_args[0][0]
        assert called_filter.status.value == "approved"


class TestGetNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/graph/some-node-id")
        assert resp.status_code == 401

    async def test_not_found(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/graph/nonexistent")
        assert resp.status_code == 404

    async def test_returns_node_with_neighbours(self, api_client):
        node = make_node()
        neighbour = make_node("n2")
        state = _make_app_state()
        state.graph_service.get_node_detail = AsyncMock(
            return_value=NodeDetailResponse(node=node, neighbours=[neighbour])
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get(f"/graph/{node.id}")
        assert resp.status_code == 200
        assert resp.json()["node"]["id"] == str(node.id)
        assert len(resp.json()["neighbours"]) == 1

    async def test_filters_non_current_neighbours(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.get_node_detail = AsyncMock(return_value=NodeDetailResponse(node=node, neighbours=[]))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get(f"/graph/{node.id}")
        assert resp.status_code == 200
        assert resp.json()["neighbours"] == []


class TestImpactTraversal:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/graph/some-node-id/impact")
        assert resp.status_code == 401

    async def test_returns_empty_when_no_neighbours(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/graph/some-node-id/impact")
        assert resp.status_code == 200
        assert resp.json()["nodes"] == []

    async def test_traverses_inbound_neighbours(self, api_client):
        node = make_node()
        neighbour = make_node("n2", confidence=0.9)
        state = _make_app_state()
        state.graph_service.traverse_impact = AsyncMock(
            return_value=ImpactResponse(nodes=[ImpactNode(node=neighbour, traversal_depth=1)])
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get(f"/graph/{node.id}/impact?depth=1")
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) == 1
        assert resp.json()["nodes"][0]["traversal_depth"] == 1

    async def test_depth_out_of_range(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/graph/some-node-id/impact?depth=10")
        assert resp.status_code == 422

    async def test_passes_args_to_service(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            await ac.get("/graph/node-1/impact?depth=3&rel_types=mitigates&min_confidence=0.5")
        call = state.graph_service.traverse_impact.call_args
        assert call.args[0] == "node-1"
        assert call.args[1] == 3
        assert call.args[2] == [RelationshipType.MITIGATES]
        assert call.args[3] == 0.5


class TestCreateNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/graph", json={"type": "decision", "title": "T", "description": "D"})
        assert resp.status_code == 401

    async def test_viewer_cannot_create(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post("/graph", json={"type": "decision", "title": "T", "description": "D"})
        assert resp.status_code == 403

    async def test_returns_201_with_node(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.create = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph", json={"type": "decision", "title": "T", "description": "D"})
        assert resp.status_code == 201
        assert resp.json()["id"] == str(node.id)

    async def test_passes_user_id_to_service(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.create = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            await ac.post("/graph", json={"type": "decision", "title": "T", "description": "D"})
        state.graph_service.create.assert_called_once()
        assert state.graph_service.create.call_args.args[1] == "alice"


class TestUpdateNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 401

    async def test_viewer_cannot_update(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 403

    async def test_returns_updated_node(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.update = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 200
        assert resp.json()["id"] == str(node.id)

    async def test_not_found_returns_404(self, api_client):
        state = _make_app_state()
        state.graph_service.update = AsyncMock(side_effect=NodeNotFoundError("node-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 404

    async def test_precondition_failure_returns_409(self, api_client):
        state = _make_app_state()
        state.graph_service.update = AsyncMock(side_effect=NodePreconditionError("not manual"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put("/graph/node-1", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 409
        assert "not manual" in resp.json()["detail"]


class TestOverrideNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        assert resp.status_code == 401

    async def test_viewer_cannot_override(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        assert resp.status_code == 403

    async def test_returns_updated_node(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.override = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        assert resp.status_code == 200
        assert resp.json()["id"] == str(node.id)

    async def test_operator_gets_auto_minimum_method(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.override = AsyncMock(return_value=node)
        async with api_client(state, make_current_user(role=UserRole.OPERATOR)) as ac:
            await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        assert state.graph_service.override.call_args.kwargs["minimum_method"] == ApprovalMethod.AUTO

    async def test_admin_gets_none_minimum_method(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.override = AsyncMock(return_value=node)
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        assert state.graph_service.override.call_args.kwargs["minimum_method"] is None

    async def test_not_found_returns_404(self, api_client):
        state = _make_app_state()
        state.graph_service.override = AsyncMock(side_effect=NodeNotFoundError("node-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        assert resp.status_code == 404

    async def test_precondition_failure_returns_409(self, api_client):
        state = _make_app_state()
        state.graph_service.override = AsyncMock(side_effect=NodePreconditionError("insufficient role"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put("/graph/node-1/override", json={"title": "T", "description": "D", "reason": "fix"})
        assert resp.status_code == 409


class TestDeleteNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.delete("/graph/node-1")
        assert resp.status_code == 401

    async def test_operator_cannot_delete(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.OPERATOR)) as ac:
            resp = await ac.delete("/graph/node-1")
        assert resp.status_code == 403

    async def test_returns_204(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.delete("/graph/node-1")
        assert resp.status_code == 204

    async def test_cascade_true_by_default(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            await ac.delete("/graph/node-1")
        state.graph_service.delete.assert_called_once()
        assert state.graph_service.delete.call_args.kwargs.get("cascade") is True

    async def test_cascade_false_when_specified(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            await ac.delete("/graph/node-1?cascade=false")
        assert state.graph_service.delete.call_args.kwargs.get("cascade") is False

    async def test_precondition_failure_returns_409(self, api_client):
        state = _make_app_state()
        state.graph_service.delete = AsyncMock(side_effect=NodePreconditionError("has inbound"))
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.delete("/graph/node-1?cascade=false")
        assert resp.status_code == 409
        assert "has inbound" in resp.json()["detail"]


class TestBulkCreateNodes:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post(
                "/graph/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        assert resp.status_code == 401

    async def test_viewer_cannot_bulk_create(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post(
                "/graph/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        assert resp.status_code == 403

    async def test_returns_bulk_result(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_create = AsyncMock(return_value=BulkResult(succeeded=["uuid-1"], failed=[]))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post(
                "/graph/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        assert resp.status_code == 200
        assert resp.json()["succeeded"] == ["uuid-1"]
        assert resp.json()["failed"] == []

    async def test_passes_user_id_to_service(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_create = AsyncMock(return_value=BulkResult(succeeded=[], failed=[]))
        async with api_client(state, make_current_user()) as ac:
            await ac.post("/graph/bulk", json={"nodes": [], "on_error": "continue"})
        state.graph_service.bulk_create.assert_called_once()
        assert state.graph_service.bulk_create.call_args.args[1] == "alice"


class TestBulkDeleteNodes:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.request("DELETE", "/graph/bulk", json={"node_ids": ["id-1"]})
        assert resp.status_code == 401

    async def test_operator_cannot_bulk_delete(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.OPERATOR)) as ac:
            resp = await ac.request("DELETE", "/graph/bulk", json={"node_ids": ["id-1"]})
        assert resp.status_code == 403

    async def test_returns_bulk_result(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_delete = AsyncMock(return_value=BulkResult(succeeded=["id-1"], failed=[]))
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.request("DELETE", "/graph/bulk", json={"node_ids": ["id-1"]})
        assert resp.status_code == 200
        assert resp.json()["succeeded"] == ["id-1"]

    async def test_cascade_passed_to_service(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_delete = AsyncMock(return_value=BulkResult(succeeded=[], failed=[]))
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            await ac.request("DELETE", "/graph/bulk?cascade=false", json={"node_ids": []})
        assert state.graph_service.bulk_delete.call_args.kwargs.get("cascade") is False

    async def test_partial_failure_in_result(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_delete = AsyncMock(
            return_value=BulkResult(
                succeeded=["id-1"],
                failed=[BulkFailure(node_id="id-2", error="not found")],
            )
        )
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.request("DELETE", "/graph/bulk", json={"node_ids": ["id-1", "id-2"]})
        assert resp.json()["failed"][0]["node_id"] == "id-2"


class TestResolveNodes:
    def _node_ids(self, *nodes):
        return [str(n.id) for n in nodes]

    async def test_requires_operator(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.resolve_by_ids = AsyncMock(return_value=0)
        async with api_client(state, make_current_user(role=UserRole.REVIEWER)) as ac:
            resp = await ac.post("/graph/nodes/resolve", json={"node_ids": self._node_ids(node)})
        assert resp.status_code == 403

    async def test_404_when_node_missing(self, api_client):
        state = _make_app_state()
        state.graph_service.resolve_by_ids = AsyncMock(side_effect=NodeNotFoundError("missing"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/nodes/resolve", json={"node_ids": ["00000000-0000-0000-0000-000000000001"]})
        assert resp.status_code == 404

    async def test_422_when_node_not_approved(self, api_client):
        state = _make_app_state()
        state.graph_service.resolve_by_ids = AsyncMock(
            side_effect=NodePreconditionError("Nodes not in APPROVED status: [...]")
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/nodes/resolve", json={"node_ids": ["00000000-0000-0000-0000-000000000001"]})
        assert resp.status_code == 422

    async def test_returns_relationship_count(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.resolve_by_ids = AsyncMock(return_value=1)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/nodes/resolve", json={"node_ids": self._node_ids(node)})
        assert resp.status_code == 200
        assert resp.json()["relationships_created"] == 1
