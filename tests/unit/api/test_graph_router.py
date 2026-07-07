from __future__ import annotations

import typing
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from seshat.app.platform.api.state import AppState
from seshat.app.services.graph import (
    NodeNotFoundError,
    NodePreconditionError,
    RelationshipConflictError,
    RelationshipNotFoundError,
)
from seshat.core.models.api_graph import BulkFailure, BulkResult
from seshat.core.models.api_responses import ImpactNode, ImpactResponse, NodeDetailResponse, NodeSearchResult
from seshat.core.models.enums import ApprovalMethod, GraphDirection, RelationshipType, SearchMode, UserRole
from tests.helpers import make_node
from tests.integration.helpers import make_relationship
from tests.unit.api.conftest import make_current_user

_NODE_ID = UUID("00000000-0000-0000-0000-000000000001")
_NODE_PATH = str(_NODE_ID)
_OVERRIDE_PAYLOAD = {"title": "T", "description": "D", "reason": "fix"}
_REL_ID = UUID("00000000-0000-0000-0000-000000000099")
_REL_PATH = str(_REL_ID)


def _make_app_state() -> AppState:
    graph_service = MagicMock()
    graph_service.query = AsyncMock(return_value=[])
    graph_service.search = AsyncMock(return_value=[])
    graph_service.get_node = AsyncMock(side_effect=NodeNotFoundError("not found"))
    graph_service.get_node_neighbours = AsyncMock(side_effect=NodeNotFoundError("not found"))
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
    graph_service.list_relationships = AsyncMock(return_value=[])
    graph_service.create_relationship = AsyncMock()
    graph_service.delete_relationship = AsyncMock()

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
            resp = await ac.get(f"/graph/{_NODE_PATH}")
        assert resp.status_code == 401

    async def test_not_found(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get(f"/graph/{_NODE_PATH}")
        assert resp.status_code == 404

    async def test_returns_node(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.get_node = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get(f"/graph/{node.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == str(node.id)


class TestGetNodeDetail:
    async def test_not_found(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get(f"/graph/{_NODE_PATH}/detail")
        assert resp.status_code == 404

    async def test_returns_node_with_neighbours(self, api_client):
        node = make_node()
        neighbour = make_node("n2")
        state = _make_app_state()
        state.graph_service.get_node_detail = AsyncMock(
            return_value=NodeDetailResponse(node=node, neighbours=[neighbour])
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get(f"/graph/{node.id}/detail")
        assert resp.status_code == 200
        assert resp.json()["node"]["id"] == str(node.id)
        assert len(resp.json()["neighbours"]) == 1

    async def test_returns_empty_neighbours(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.get_node_detail = AsyncMock(return_value=NodeDetailResponse(node=node, neighbours=[]))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get(f"/graph/{node.id}/detail")
        assert resp.status_code == 200
        assert resp.json()["neighbours"] == []


class TestImpactTraversal:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get(f"/graph/{_NODE_PATH}/impact")
        assert resp.status_code == 401

    async def test_returns_empty_when_no_neighbours(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get(f"/graph/{_NODE_PATH}/impact")
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
            resp = await ac.get(f"/graph/{_NODE_PATH}/impact?depth=10")
        assert resp.status_code == 422

    async def test_passes_args_to_service(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            await ac.get(f"/graph/{_NODE_PATH}/impact?depth=3&rel_types=mitigates&min_confidence=0.5")
        call = state.graph_service.traverse_impact.call_args
        assert call.args[0] == _NODE_ID
        assert call.args[1] == 3
        assert call.args[2] == [RelationshipType.MITIGATES]
        assert call.args[3] == 0.5
        assert call.args[4] == GraphDirection.OUTBOUND

    async def test_direction_inbound_forwarded_to_service(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            await ac.get(f"/graph/{_NODE_PATH}/impact?direction=inbound")
        call = state.graph_service.traverse_impact.call_args
        assert call.args[4] == GraphDirection.INBOUND


class TestGetNodeNeighbours:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get(f"/graph/{_NODE_PATH}/neighbours")
        assert resp.status_code == 401

    async def test_not_found(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get(f"/graph/{_NODE_PATH}/neighbours")
        assert resp.status_code == 404

    async def test_returns_neighbours(self, api_client):
        neighbour = make_node("n2")
        state = _make_app_state()
        state.graph_service.get_node_neighbours = AsyncMock(return_value=[neighbour])
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get(f"/graph/{_NODE_PATH}/neighbours")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["id"] == str(neighbour.id)


class TestCreateNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/graph/nodes", json={"type": "decision", "title": "T", "description": "D"})
        assert resp.status_code == 401

    async def test_viewer_cannot_create(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post("/graph/nodes", json={"type": "decision", "title": "T", "description": "D"})
        assert resp.status_code == 403

    async def test_returns_201_with_node(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.create = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/nodes", json={"type": "decision", "title": "T", "description": "D"})
        assert resp.status_code == 201
        assert resp.json()["id"] == str(node.id)

    async def test_passes_user_id_to_service(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.create = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            await ac.post("/graph/nodes", json={"type": "decision", "title": "T", "description": "D"})
        state.graph_service.create.assert_called_once()
        assert state.graph_service.create.call_args.args[1] == "alice"


class TestUpdateNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 401

    async def test_viewer_cannot_update(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 403

    async def test_returns_updated_node(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.update = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 200
        assert resp.json()["id"] == str(node.id)

    async def test_not_found_returns_404(self, api_client):
        state = _make_app_state()
        state.graph_service.update = AsyncMock(side_effect=NodeNotFoundError(_NODE_PATH))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 404

    async def test_precondition_failure_returns_409(self, api_client):
        state = _make_app_state()
        state.graph_service.update = AsyncMock(side_effect=NodePreconditionError("not manual"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}", json={"title": "T", "description": "D", "reason": None})
        assert resp.status_code == 409
        assert "not manual" in resp.json()["detail"]


class TestOverrideNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}/override", json=_OVERRIDE_PAYLOAD)
        assert resp.status_code == 401

    async def test_viewer_cannot_override(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}/override", json=_OVERRIDE_PAYLOAD)
        assert resp.status_code == 403

    async def test_returns_updated_node(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.override = AsyncMock(return_value=node)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}/override", json=_OVERRIDE_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["id"] == str(node.id)

    async def test_operator_gets_auto_minimum_method(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.override = AsyncMock(return_value=node)
        async with api_client(state, make_current_user(role=UserRole.OPERATOR)) as ac:
            await ac.put(f"/graph/nodes/{_NODE_PATH}/override", json=_OVERRIDE_PAYLOAD)
        assert state.graph_service.override.call_args.kwargs["minimum_method"] == ApprovalMethod.AUTO

    async def test_admin_gets_none_minimum_method(self, api_client):
        node = make_node()
        state = _make_app_state()
        state.graph_service.override = AsyncMock(return_value=node)
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            await ac.put(f"/graph/nodes/{_NODE_PATH}/override", json=_OVERRIDE_PAYLOAD)
        assert state.graph_service.override.call_args.kwargs["minimum_method"] is None

    async def test_not_found_returns_404(self, api_client):
        state = _make_app_state()
        state.graph_service.override = AsyncMock(side_effect=NodeNotFoundError(_NODE_PATH))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}/override", json=_OVERRIDE_PAYLOAD)
        assert resp.status_code == 404

    async def test_precondition_failure_returns_409(self, api_client):
        state = _make_app_state()
        state.graph_service.override = AsyncMock(side_effect=NodePreconditionError("insufficient role"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.put(f"/graph/nodes/{_NODE_PATH}/override", json=_OVERRIDE_PAYLOAD)
        assert resp.status_code == 409


class TestDeleteNode:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.delete(f"/graph/nodes/{_NODE_PATH}")
        assert resp.status_code == 401

    async def test_operator_cannot_delete(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.OPERATOR)) as ac:
            resp = await ac.delete(f"/graph/nodes/{_NODE_PATH}")
        assert resp.status_code == 403

    async def test_returns_204(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.delete(f"/graph/nodes/{_NODE_PATH}")
        assert resp.status_code == 204

    async def test_cascade_true_by_default(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            await ac.delete(f"/graph/nodes/{_NODE_PATH}")
        state.graph_service.delete.assert_called_once()
        assert state.graph_service.delete.call_args.kwargs.get("cascade") is True

    async def test_cascade_false_when_specified(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            await ac.delete(f"/graph/nodes/{_NODE_PATH}?cascade=false")
        assert state.graph_service.delete.call_args.kwargs.get("cascade") is False

    async def test_precondition_failure_returns_409(self, api_client):
        state = _make_app_state()
        state.graph_service.delete = AsyncMock(side_effect=NodePreconditionError("has inbound"))
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.delete(f"/graph/nodes/{_NODE_PATH}?cascade=false")
        assert resp.status_code == 409
        assert "has inbound" in resp.json()["detail"]


class TestBulkCreateNodes:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post(
                "/graph/nodes/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        assert resp.status_code == 401

    async def test_viewer_cannot_bulk_create(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post(
                "/graph/nodes/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        assert resp.status_code == 403

    async def test_returns_bulk_result(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_create = AsyncMock(return_value=BulkResult(succeeded=["uuid-1"], failed=[]))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post(
                "/graph/nodes/bulk", json={"nodes": [{"type": "decision", "title": "T", "description": "D"}]}
            )
        assert resp.status_code == 200
        assert resp.json()["succeeded"] == ["uuid-1"]
        assert resp.json()["failed"] == []

    async def test_passes_user_id_to_service(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_create = AsyncMock(return_value=BulkResult(succeeded=[], failed=[]))
        async with api_client(state, make_current_user()) as ac:
            await ac.post("/graph/nodes/bulk", json={"nodes": [], "on_error": "continue"})
        state.graph_service.bulk_create.assert_called_once()
        assert state.graph_service.bulk_create.call_args.args[1] == "alice"


class TestBulkDeleteNodes:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.request("DELETE", "/graph/nodes/bulk", json={"node_ids": [_NODE_PATH]})
        assert resp.status_code == 401

    async def test_operator_cannot_bulk_delete(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.OPERATOR)) as ac:
            resp = await ac.request("DELETE", "/graph/nodes/bulk", json={"node_ids": [_NODE_PATH]})
        assert resp.status_code == 403

    async def test_returns_bulk_result(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_delete = AsyncMock(return_value=BulkResult(succeeded=[_NODE_PATH], failed=[]))
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.request("DELETE", "/graph/nodes/bulk", json={"node_ids": [_NODE_PATH]})
        assert resp.status_code == 200
        assert resp.json()["succeeded"] == [_NODE_PATH]

    async def test_cascade_passed_to_service(self, api_client):
        state = _make_app_state()
        state.graph_service.bulk_delete = AsyncMock(return_value=BulkResult(succeeded=[], failed=[]))
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            await ac.request("DELETE", "/graph/nodes/bulk?cascade=false", json={"node_ids": []})
        assert state.graph_service.bulk_delete.call_args.kwargs.get("cascade") is False

    async def test_partial_failure_in_result(self, api_client):
        _node_id_2 = str(UUID("00000000-0000-0000-0000-000000000002"))
        state = _make_app_state()
        state.graph_service.bulk_delete = AsyncMock(
            return_value=BulkResult(
                succeeded=[_NODE_PATH],
                failed=[BulkFailure(node_id=_node_id_2, error="not found")],
            )
        )
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.request("DELETE", "/graph/nodes/bulk", json={"node_ids": [_NODE_PATH, _node_id_2]})
        assert resp.json()["failed"][0]["node_id"] == _node_id_2


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
            resp = await ac.post("/graph/nodes/resolve", json={"node_ids": [_NODE_PATH]})
        assert resp.status_code == 404

    async def test_422_when_node_not_approved(self, api_client):
        state = _make_app_state()
        state.graph_service.resolve_by_ids = AsyncMock(
            side_effect=NodePreconditionError("Nodes not in APPROVED status: [...]")
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/nodes/resolve", json={"node_ids": [_NODE_PATH]})
        assert resp.status_code == 422

    async def test_returns_relationship_count(self, api_client):
        node = make_node()
        rel = make_relationship(node, make_node("tgt"))
        state = _make_app_state()
        state.graph_service.resolve_by_ids = AsyncMock(return_value=[rel])
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/nodes/resolve", json={"node_ids": self._node_ids(node)})
        assert resp.status_code == 200
        assert len(resp.json()["relationships_created"]) == 1


class TestSearchGraph:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/graph/search", params={"q": "auth risk"})
        assert resp.status_code == 401

    async def test_returns_empty_results(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/graph/search", params={"q": "auth risk"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    async def test_returns_matching_results(self, api_client):
        node = make_node()
        detail = NodeDetailResponse(node=node, neighbours=[])
        state = _make_app_state()
        state.graph_service.search = AsyncMock(return_value=[NodeSearchResult(detail=detail)])
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/graph/search", params={"q": "auth risk"})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1

    async def test_missing_q_returns_422(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/graph/search")
        assert resp.status_code == 422

    async def test_search_mode_forwarded_to_service(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/graph/search", params={"q": "auth risk", "search_mode": "keyword"})
        assert resp.status_code == 200
        _, kwargs = state.graph_service.search.call_args
        assert kwargs["mode"] == SearchMode.KEYWORD


class TestListRelationships:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/graph/relationships")
        assert resp.status_code == 401

    async def test_returns_empty_list(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/graph/relationships")
        assert resp.status_code == 200
        assert resp.json()["relationships"] == []

    async def test_returns_relationships(self, api_client):
        node = make_node()
        rel = make_relationship(node, make_node("n2"))
        state = _make_app_state()
        state.graph_service.list_relationships = AsyncMock(return_value=[rel])
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/graph/relationships")
        assert resp.status_code == 200
        assert len(resp.json()["relationships"]) == 1

    async def test_passes_filters_to_service(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            await ac.get(f"/graph/relationships?node_id={_NODE_PATH}&rel_type=supersedes&limit=50")
        call = state.graph_service.list_relationships.call_args
        assert call.kwargs["node_id"] == _NODE_ID
        assert call.kwargs["rel_type"] == RelationshipType.SUPERSEDES
        assert call.kwargs["limit"] == 50


class TestCreateRelationship:
    _payload: typing.ClassVar = {
        "source_id": str(UUID("00000000-0000-0000-0000-000000000001")),
        "target_id": str(UUID("00000000-0000-0000-0000-000000000002")),
        "rel_type": "supersedes",
    }

    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/graph/relationships", json=self._payload)
        assert resp.status_code == 401

    async def test_viewer_cannot_create(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post("/graph/relationships", json=self._payload)
        assert resp.status_code == 403

    async def test_returns_201_with_relationship(self, api_client):
        node = make_node()
        rel = make_relationship(node, make_node("n2"))
        state = _make_app_state()
        state.graph_service.create_relationship = AsyncMock(return_value=rel)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/relationships", json=self._payload)
        assert resp.status_code == 201
        assert "rel_id" in resp.json()

    async def test_404_when_node_not_found(self, api_client):
        state = _make_app_state()
        state.graph_service.create_relationship = AsyncMock(side_effect=NodeNotFoundError("missing"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/relationships", json=self._payload)
        assert resp.status_code == 404

    async def test_409_on_duplicate(self, api_client):
        state = _make_app_state()
        state.graph_service.create_relationship = AsyncMock(side_effect=RelationshipConflictError("already exists"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/graph/relationships", json=self._payload)
        assert resp.status_code == 409


class TestDeleteRelationship:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.delete(f"/graph/relationships/{_REL_PATH}")
        assert resp.status_code == 401

    async def test_operator_cannot_delete(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.OPERATOR)) as ac:
            resp = await ac.delete(f"/graph/relationships/{_REL_PATH}")
        assert resp.status_code == 403

    async def test_returns_204(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.delete(f"/graph/relationships/{_REL_PATH}")
        assert resp.status_code == 204

    async def test_404_when_not_found(self, api_client):
        state = _make_app_state()
        state.graph_service.delete_relationship = AsyncMock(side_effect=RelationshipNotFoundError("not found"))
        async with api_client(state, make_current_user(role=UserRole.ADMIN)) as ac:
            resp = await ac.delete(f"/graph/relationships/{_REL_PATH}")
        assert resp.status_code == 404
