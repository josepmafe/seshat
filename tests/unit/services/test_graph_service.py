from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from seshat.models.api_graph import (
    BulkNodeCreate,
    BulkNodeDelete,
    ManualNodeCreate,
    ManualNodeUpdate,
    NodeOverride,
    RelationshipInput,
)
from seshat.models.enums import (
    ApprovalMethod,
    ConceptType,
    IngestionSource,
    NodeState,
    NodeStatus,
    RelationshipType,
    SearchMode,
)
from seshat.models.nodes import KBNode, NodeMetadata, ResolutionResult
from seshat.services.graph_service import GraphService, NodeNotFoundError, NodePreconditionError
from tests.helpers import make_node
from tests.integration.helpers import make_relationship

_UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
_UUID_2 = UUID("00000000-0000-0000-0000-000000000002")


def _make_service(*, node: KBNode | None = None, inbound_count: int = 0):
    repo = MagicMock()
    repo.get_node = AsyncMock(return_value=node)
    repo.get_neighbours = AsyncMock(return_value=[])
    repo.write_node = AsyncMock()
    repo.write_relationship = AsyncMock()
    repo.update_node = AsyncMock()
    repo.delete_node = AsyncMock()
    repo.count_inbound_relationships = AsyncMock(return_value=inbound_count)

    extraction_orch = MagicMock()
    extraction_orch.run_resolution = AsyncMock(return_value=ResolutionResult(job_id="job-1", relationships=[]))

    return GraphService(repo, extraction_orch), repo


def _manual_metadata() -> NodeMetadata:
    return NodeMetadata(
        job_id="manual_abc",
        ingestion_source=IngestionSource.MANUAL,
        approval_method=ApprovalMethod.MANUAL,
    )


def _auto_metadata() -> NodeMetadata:
    return NodeMetadata(job_id="job-1", ingestion_source=IngestionSource.JOB, approval_method=ApprovalMethod.AUTO)


def _create_payload(
    title: str = "T",
    description: str = "D",
    relationships=None,
    source_quote: str | None = None,
    blob_key: str | None = None,
) -> ManualNodeCreate:
    return ManualNodeCreate(
        type=ConceptType.DECISION,
        title=title,
        description=description,
        relationships=relationships,
        source_quote=source_quote,
        blob_key=blob_key,
    )


def _update_payload(
    title: str = "T2",
    description: str = "D2",
    reason: str | None = None,
    relationships=None,
) -> ManualNodeUpdate:
    return ManualNodeUpdate(title=title, description=description, reason=reason, relationships=relationships)


def _override_payload(
    title: str = "T3",
    description: str = "D3",
    reason: str = "Correction",
    relationships=None,
) -> NodeOverride:
    return NodeOverride(title=title, description=description, reason=reason, relationships=relationships)


class TestCreate:
    async def test_returns_kb_node(self):
        svc, _ = _make_service()
        node = await svc.create(_create_payload(), user_id="alice")

        assert isinstance(node, KBNode)
        assert node.status == NodeStatus.APPROVED
        assert node.state == NodeState.CURRENT
        assert node.metadata.ingestion_source == IngestionSource.MANUAL
        assert node.metadata.approval_method == ApprovalMethod.MANUAL
        assert node.metadata.approved_by == "alice"

    async def test_job_id_has_manual_prefix(self):
        svc, _ = _make_service()
        node = await svc.create(_create_payload(), user_id="alice")
        assert node.metadata.job_id.startswith("manual_")

    async def test_job_id_is_unique_across_creates(self):
        svc, _ = _make_service()
        n1 = await svc.create(_create_payload(), user_id="alice")
        n2 = await svc.create(_create_payload(), user_id="alice")
        assert n1.metadata.job_id != n2.metadata.job_id

    async def test_writes_to_repo(self):
        svc, repo = _make_service()
        await svc.create(_create_payload(), user_id="alice")
        repo.write_node.assert_called_once()

    async def test_writes_relationships_when_provided(self):
        target = make_node("tgt")
        svc, repo = _make_service()
        rel = RelationshipInput(target_id=str(target.id), rel_type=RelationshipType.SUPERSEDES)
        await svc.create(_create_payload(relationships=[rel]), user_id="alice")

        call_kwargs = repo.write_node.call_args.kwargs
        rels = call_kwargs.get("relationships") or []
        assert len(rels) == 1
        assert rels[0].rel_type == RelationshipType.SUPERSEDES
        assert rels[0].target_id == target.id

    async def test_skips_relationships_when_none(self):
        svc, repo = _make_service()
        await svc.create(_create_payload(relationships=None), user_id="alice")
        call_kwargs = repo.write_node.call_args.kwargs
        assert call_kwargs.get("relationships") is None

    async def test_logs_warning_for_source_quote(self, caplog):
        svc, _ = _make_service()
        payload = _create_payload(source_quote="some quote", blob_key="blobs/key")
        with caplog.at_level(logging.WARNING, logger="seshat.services.graph_service"):
            await svc.create(payload, user_id="alice")
        assert "not yet implemented" in caplog.text


class TestUpdate:
    async def test_raises_not_found_when_node_missing(self):
        svc, _ = _make_service(node=None)
        with pytest.raises(NodeNotFoundError):
            await svc.update("missing-id", _update_payload(), user_id="alice")

    async def test_raises_precondition_for_non_manual_node(self):
        node = make_node()
        svc, _ = _make_service(node=node)
        with pytest.raises(NodePreconditionError, match="manually-created"):
            await svc.update(str(node.id), _update_payload(), user_id="alice")

    async def test_updates_manual_node(self):
        node = make_node(metadata=_manual_metadata())
        svc, repo = _make_service(node=node)
        result = await svc.update(str(node.id), _update_payload(title="New Title"), user_id="alice")

        assert result.title == "New Title"
        assert result.metadata.corrected_by == "alice"
        repo.update_node.assert_called_once()

    async def test_replaces_relationships_when_provided(self):
        target = make_node("tgt")
        node = make_node(metadata=_manual_metadata())
        svc, repo = _make_service(node=node)
        rel = RelationshipInput(target_id=str(target.id), rel_type=RelationshipType.SUPERSEDES)
        await svc.update(str(node.id), _update_payload(relationships=[rel]), user_id="alice")

        call_kwargs = repo.update_node.call_args.kwargs
        assert call_kwargs.get("replace_outbound_rels") is True
        rels = call_kwargs.get("relationships") or []
        assert len(rels) == 1
        assert rels[0].rel_type == RelationshipType.SUPERSEDES
        assert rels[0].target_id == target.id

    async def test_preserves_relationships_when_none(self):
        node = make_node(metadata=_manual_metadata())
        svc, repo = _make_service(node=node)
        await svc.update(str(node.id), _update_payload(relationships=None), user_id="alice")

        call_kwargs = repo.update_node.call_args.kwargs
        assert call_kwargs.get("replace_outbound_rels") is False
        assert call_kwargs.get("relationships") is None


class TestOverride:
    async def test_raises_not_found_when_node_missing(self):
        svc, _ = _make_service(node=None)
        with pytest.raises(NodeNotFoundError):
            await svc.override("missing-id", _override_payload(), user_id="alice", minimum_method=None)

    async def test_raises_precondition_when_method_does_not_match(self):
        node = make_node()
        svc, _ = _make_service(node=node)
        with pytest.raises(NodePreconditionError, match="Insufficient role"):
            await svc.override(str(node.id), _override_payload(), user_id="alice", minimum_method=ApprovalMethod.AUTO)

    async def test_allows_when_method_matches_exactly(self):
        node = make_node(metadata=_manual_metadata())
        svc, repo = _make_service(node=node)
        await svc.override(str(node.id), _override_payload(), user_id="alice", minimum_method=ApprovalMethod.MANUAL)
        repo.update_node.assert_called_once()

    async def test_allows_any_node_when_minimum_method_is_none(self):
        node = make_node()
        svc, repo = _make_service(node=node)
        result = await svc.override(
            str(node.id),
            _override_payload(reason="Admin fix"),
            user_id="admin",
            minimum_method=None,
        )

        assert result.metadata.correction_reason == "Admin fix"
        repo.update_node.assert_called_once()

    async def test_stores_reason_and_corrected_by_in_metadata(self):
        node = make_node(metadata=_auto_metadata())
        svc, _ = _make_service(node=node)
        result = await svc.override(
            str(node.id),
            _override_payload(reason="Wrong decision"),
            user_id="alice",
            minimum_method=ApprovalMethod.AUTO,
        )

        assert result.metadata.correction_reason == "Wrong decision"
        assert result.metadata.corrected_by == "alice"


class TestDelete:
    async def test_cascade_deletes_node(self):
        svc, repo = _make_service()
        await svc.delete("node-id", cascade=True)

        repo.delete_node.assert_called_once_with("node-id", cascade=True)

    async def test_safe_delete_passes_when_no_inbound(self):
        svc, repo = _make_service(inbound_count=0)
        await svc.delete("node-id", cascade=False)

        repo.delete_node.assert_called_once_with("node-id", cascade=False)

    async def test_safe_delete_raises_when_inbound_relationships_exist(self):
        svc, repo = _make_service(inbound_count=3)
        with pytest.raises(NodePreconditionError, match="referenced as a target by 3"):
            await svc.delete("node-id", cascade=False)

        repo.delete_node.assert_not_called()


class TestBulkCreate:
    async def test_returns_succeeded_ids(self):
        svc, _ = _make_service()
        payload = BulkNodeCreate(nodes=[_create_payload(), _create_payload(title="T2")])
        result = await svc.bulk_create(payload, user_id="alice")

        assert len(result.succeeded) == 2
        assert result.failed == []

    async def test_stop_on_error_propagates_exception(self):
        svc, repo = _make_service()
        repo.write_node = AsyncMock(side_effect=RuntimeError("db down"))
        payload = BulkNodeCreate(nodes=[_create_payload()], on_error="stop")

        with pytest.raises(RuntimeError, match="db down"):
            await svc.bulk_create(payload, user_id="alice")

    async def test_continue_on_error_collects_failures(self):
        svc, repo = _make_service()
        repo.write_node = AsyncMock(side_effect=RuntimeError("db down"))
        payload = BulkNodeCreate(nodes=[_create_payload(), _create_payload(title="T2")], on_error="continue")
        result = await svc.bulk_create(payload, user_id="alice")

        assert result.succeeded == []
        assert len(result.failed) == 2
        assert "db down" in result.failed[0].error


_UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
_UUID_2 = UUID("00000000-0000-0000-0000-000000000002")


class TestBulkDelete:
    async def test_returns_succeeded_ids(self):
        svc, _ = _make_service()
        payload = BulkNodeDelete(node_ids=[_UUID_1, _UUID_2])
        result = await svc.bulk_delete(payload)

        assert result.succeeded == [str(_UUID_1), str(_UUID_2)]
        assert result.failed == []

    async def test_stop_on_error_propagates_exception(self):
        svc, _ = _make_service(inbound_count=1)
        payload = BulkNodeDelete(node_ids=[_UUID_1], on_error="stop")

        with pytest.raises(NodePreconditionError):
            await svc.bulk_delete(payload, cascade=False)

    async def test_continue_on_error_collects_failures(self):
        svc, _ = _make_service(inbound_count=1)
        payload = BulkNodeDelete(node_ids=[_UUID_1, _UUID_2], on_error="continue")
        result = await svc.bulk_delete(payload, cascade=False)

        assert result.succeeded == []
        assert len(result.failed) == 2
        assert result.failed[0].node_id == str(_UUID_1)


class TestGetNode:
    async def test_raises_not_found_when_missing(self):
        svc, _ = _make_service(node=None)
        with pytest.raises(NodeNotFoundError):
            await svc.get_node(_UUID_1)

    async def test_returns_node_when_present(self):
        node = make_node()
        svc, _ = _make_service(node=node)
        result = await svc.get_node(node.id)
        assert result == node


class TestGetNodeNeighbours:
    async def test_raises_not_found_when_node_missing(self):
        svc, _ = _make_service(node=None)
        with pytest.raises(NodeNotFoundError):
            await svc.get_node_neighbours(_UUID_1)

    async def test_returns_current_neighbours(self):
        node = make_node()
        neighbour = make_node("n2")
        svc, repo = _make_service(node=node)
        repo.get_neighbours = AsyncMock(return_value=[neighbour])
        result = await svc.get_node_neighbours(node.id)
        assert len(result) == 1
        assert result[0].id == neighbour.id


class TestGetNodeDetail:
    async def test_raises_not_found_when_missing(self):
        svc, _ = _make_service(node=None)
        with pytest.raises(NodeNotFoundError):
            await svc.get_node_detail(_UUID_1)

    async def test_returns_detail_with_current_neighbours_only(self):
        node = make_node()
        current_neighbour = make_node("n2")
        superseded_neighbour = make_node("n3")
        superseded_neighbour = superseded_neighbour._with(state=NodeState.SUPERSEDED)

        svc, repo = _make_service(node=node)
        repo.get_neighbours = AsyncMock(return_value=[current_neighbour, superseded_neighbour])

        detail = await svc.get_node_detail(node.id)

        assert detail.node == node
        assert len(detail.neighbours) == 1
        assert detail.neighbours[0].id == current_neighbour.id

    async def test_no_neighbours_returns_empty_list(self):
        node = make_node()
        svc, repo = _make_service(node=node)
        repo.get_neighbours = AsyncMock(return_value=[])

        detail = await svc.get_node_detail(node.id)

        assert detail.neighbours == []

    async def test_relationships_field_populated(self):
        node = make_node()
        rel = make_relationship(node, make_node("tgt"))
        svc, repo = _make_service(node=node)
        repo.get_node_relationships = AsyncMock(return_value=[rel])

        detail = await svc.get_node_detail(node.id)

        assert len(detail.relationships) == 1
        assert detail.relationships[0].source_id == node.id


class TestTraverseImpact:
    async def test_returns_empty_when_no_inbound_neighbours(self):
        svc, repo = _make_service()
        repo.get_neighbours = AsyncMock(return_value=[])

        result = await svc.traverse_impact(_UUID_1, depth=1, rel_types=None, min_confidence=0.0)

        assert result.nodes == []

    async def test_single_hop_returns_inbound_neighbour(self):
        neighbour = make_node("n2", confidence=0.8)
        svc, repo = _make_service(node=neighbour)
        repo.get_neighbours = AsyncMock(return_value=[neighbour])

        result = await svc.traverse_impact(_UUID_1, depth=1, rel_types=None, min_confidence=0.0)

        assert len(result.nodes) == 1
        assert result.nodes[0].node.id == neighbour.id
        assert result.nodes[0].traversal_depth == 1

    async def test_skips_nodes_below_min_confidence(self):
        low_conf = make_node("low", confidence=0.3)
        svc, repo = _make_service(node=low_conf)
        repo.get_neighbours = AsyncMock(return_value=[low_conf])

        result = await svc.traverse_impact(_UUID_1, depth=1, rel_types=None, min_confidence=0.5)

        assert result.nodes == []

    async def test_does_not_revisit_already_seen_nodes(self):
        node_a = make_node("a")
        svc, repo = _make_service(node=node_a)

        call_count = 0

        async def _get_neighbours(nid, *, rel_types, direction):
            nonlocal call_count
            call_count += 1
            # Return node_a for both hops, but it should only appear once.
            return [node_a] if nid == _UUID_1 else []

        repo.get_neighbours = _get_neighbours
        repo.get_node = AsyncMock(return_value=node_a)

        result = await svc.traverse_impact(_UUID_1, depth=2, rel_types=None, min_confidence=0.0)

        assert len(result.nodes) == 1

    async def test_relationships_included_in_response(self):
        neighbour = make_node("n2", confidence=0.8)
        rel = make_relationship(make_node(), neighbour)
        svc, repo = _make_service(node=neighbour)
        repo.get_neighbours = AsyncMock(return_value=[neighbour])
        repo.get_node_relationships = AsyncMock(return_value=[rel])

        result = await svc.traverse_impact(_UUID_1, depth=1, rel_types=None, min_confidence=0.0)

        assert len(result.relationships) == 1

    async def test_forwards_direction_to_repo(self):
        node_a = make_node("a")
        svc, repo = _make_service(node=node_a)
        captured: list[GraphDirection] = []

        async def _get_neighbours(nid, *, rel_types, direction):
            captured.append(direction)
            return [node_a] if nid == _UUID_1 else []

        repo.get_neighbours = _get_neighbours
        repo.get_node = AsyncMock(return_value=node_a)

        await svc.traverse_impact(
            _UUID_1, depth=1, rel_types=None, min_confidence=0.0, direction=GraphDirection.INBOUND
        )

        assert captured == [GraphDirection.INBOUND]


class TestResolveByIds:
    async def test_raises_not_found_for_missing_node(self):
        svc, _repo = _make_service(node=None)
        with pytest.raises(NodeNotFoundError):
            await svc.resolve_by_ids([_UUID_1])

    async def test_raises_precondition_for_non_approved_node(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        svc, _repo = _make_service(node=node)
        with pytest.raises(NodePreconditionError, match="APPROVED"):
            await svc.resolve_by_ids([node.id])

    async def test_returns_count_of_relationships(self):
        node = make_node()
        rel = make_relationship(node, make_node("tgt"))
        svc, repo = _make_service(node=node)
        repo.get_node = AsyncMock(return_value=node)
        svc._extraction_orch.run_resolution = AsyncMock(return_value=ResolutionResult(job_id="x", relationships=[rel]))

        rels = await svc.resolve_by_ids([node.id])

        assert len(rels) == 1

    async def test_writes_each_relationship(self):
        node = make_node()
        rel = make_relationship(node, make_node("tgt"))
        svc, repo = _make_service(node=node)
        svc._extraction_orch.run_resolution = AsyncMock(return_value=ResolutionResult(job_id="x", relationships=[rel]))

        await svc.resolve_by_ids([node.id])

        repo.write_relationship.assert_called_once_with(rel)


class TestGetNodeDetailNeighbourFiltering:
    async def test_superseded_source_node_excludes_all_neighbours(self):
        """A SUPERSEDED source node has no current neighbours in the response."""
        node = make_node()
        node = node._with(state=NodeState.SUPERSEDED)
        neighbour = make_node("n2")

        svc, repo = _make_service(node=node)
        repo.get_neighbours = AsyncMock(return_value=[neighbour])

        detail = await svc.get_node_detail(node.id)

        assert detail.neighbours == []


class TestTraverseImpactMultiHop:
    async def test_two_hop_traversal_assigns_correct_depth(self):
        node_a = make_node("a")
        node_b = make_node("b")
        svc, repo = _make_service()

        async def _get_neighbours(nid, *, rel_types, direction):
            if nid == _UUID_1:
                return [node_a]
            if nid == node_a.id:
                return [node_b]
            return []

        repo.get_neighbours = _get_neighbours

        async def _get_node(nid):
            return {node_a.id: node_a, node_b.id: node_b}.get(nid)

        repo.get_node = _get_node

        result = await svc.traverse_impact(_UUID_1, depth=2, rel_types=None, min_confidence=0.0)

        depths = {r.node.id: r.traversal_depth for r in result.nodes}
        assert depths[node_a.id] == 1
        assert depths[node_b.id] == 2


class TestBulkDeleteFailed:
    async def test_failed_ids_contain_node_id_strings(self):
        svc, _ = _make_service(inbound_count=1)
        payload = BulkNodeDelete(node_ids=[_UUID_1], on_error="continue")
        result = await svc.bulk_delete(payload, cascade=False)

        assert result.failed[0].node_id == str(_UUID_1)


class TestBulkCreatePartialFailure:
    async def test_succeeded_list_contains_only_written_ids(self):
        written = []
        write_calls = 0

        svc, repo = _make_service()

        async def _write_node(node, *, relationships=None):
            nonlocal write_calls
            write_calls += 1
            if write_calls > 1:
                raise RuntimeError("db error")
            written.append(node.id)

        repo.write_node = _write_node
        payload = BulkNodeCreate(nodes=[_create_payload(), _create_payload(title="T2")], on_error="continue")
        result = await svc.bulk_create(payload, user_id="alice")

        assert len(result.succeeded) == 1
        assert len(result.failed) == 1


class TestSearch:
    async def test_returns_node_details_for_results(self):
        node = make_node()
        result = MagicMock()
        result.node_id = node.id
        svc, repo = _make_service(node=node)
        repo.search = AsyncMock(return_value=[result])
        repo.get_neighbours = AsyncMock(return_value=[])

        from seshat.models.api_graph import NodeFilter

        details = await svc.search("auth risk", limit=5, node_filter=NodeFilter())

        assert len(details) == 1
        assert details[0].detail.node.id == node.id

    async def test_skips_missing_nodes(self):
        result = MagicMock()
        result.node_id = _UUID_1
        svc, repo = _make_service(node=None)
        repo.search = AsyncMock(return_value=[result])

        from seshat.models.api_graph import NodeFilter

        details = await svc.search("auth risk", limit=5, node_filter=NodeFilter())

        assert details == []

    async def test_returns_empty_for_no_results(self):
        svc, repo = _make_service()
        repo.search = AsyncMock(return_value=[])

        from seshat.models.api_graph import NodeFilter

        details = await svc.search("nothing", limit=10, node_filter=NodeFilter())

        assert details == []

    async def test_mode_forwarded_to_repo(self):
        svc, repo = _make_service()
        repo.search = AsyncMock(return_value=[])

        from seshat.models.api_graph import NodeFilter

        await svc.search("q", limit=5, node_filter=NodeFilter(), mode=SearchMode.KEYWORD)

        _, kwargs = repo.search.call_args
        assert kwargs["mode"] == SearchMode.KEYWORD
