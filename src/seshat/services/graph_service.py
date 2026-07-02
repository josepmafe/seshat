from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from seshat.models.api_graph import BulkFailure, BulkNodeCreate, BulkNodeDelete, BulkResult
from seshat.models.api_responses import ImpactNode, ImpactResponse, NodeDetailResponse
from seshat.models.enums import ApprovalMethod, GraphDirection, IngestionSource, NodeState, NodeStatus, RelationshipType
from seshat.models.nodes import KBNode, KBRelationship, NodeMetadata
from seshat.observability.latency_tracker import track_latency_profile
from seshat.observability.usage_tracker import UsageTracker, track_token_budget
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.models.api_graph import ManualNodeCreate, ManualNodeUpdate, NodeFilter, NodeOverride, RelationshipInput
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.repositories.node_repository import NodeRepository

logger = get_logger(__name__)


class NodeNotFoundError(Exception):
    pass


class NodePreconditionError(Exception):
    pass


class GraphService:
    def __init__(
        self,
        node_repo: NodeRepository,
        extraction_orch: ExtractionOrchestrator,
    ) -> None:
        self._repo = node_repo
        self._extraction_orch = extraction_orch
        self._usage_tracker = UsageTracker.uncapped()

    # -- Read methods ----------------------------------------------------------

    async def query(self, node_filter: NodeFilter) -> list[KBNode]:
        return await self._repo.query(node_filter)

    async def get_node_detail(self, node_id: UUID) -> NodeDetailResponse:
        node = await self._repo.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)

        neighbours = await self._repo.get_neighbours(node_id, direction=GraphDirection.BOTH)
        active_neighbours = [n for n in neighbours if _both_current(node, n)]
        return NodeDetailResponse(node=node, neighbours=active_neighbours)

    async def traverse_impact(
        self,
        node_id: UUID,
        depth: int,
        rel_types: list[RelationshipType] | None,
        min_confidence: float,
    ) -> ImpactResponse:
        visited: dict[UUID, int] = {}
        frontier = [node_id]

        for hop in range(1, depth + 1):
            next_frontier = []
            for nid in frontier:
                neighbours = await self._repo.get_neighbours(nid, rel_types=rel_types, direction=GraphDirection.INBOUND)
                for n in neighbours:
                    if n.id not in visited and n.confidence >= min_confidence:
                        visited[n.id] = hop
                        next_frontier.append(n.id)
            frontier = next_frontier

        impact_nodes: list[ImpactNode] = []
        for nid, hop in visited.items():
            n = await self._repo.get_node(nid)
            if n:
                impact_nodes.append(ImpactNode(node=n, traversal_depth=hop))

        return ImpactResponse(nodes=impact_nodes)

    # -- Write methods ---------------------------------------------------------

    @track_token_budget("manual_node_create", uncapped=True, accumulate_to_fn=lambda self: self._usage_tracker)
    async def create(self, payload: ManualNodeCreate, user_id: str) -> KBNode:
        now = datetime.now(UTC)
        job_id = f"manual_{uuid4()}"

        if payload.source_quote is not None:
            logger.warning("blob-based quote anchors are not yet implemented — ignoring source_quote")

        node = _build_manual_node(job_id, payload, now, user_id)
        relationships = _build_relationships(node.id, payload.relationships or [], now, job_id=job_id)
        await self._repo.write_node(node, relationships=relationships or None)

        return node

    async def bulk_create(self, payload: BulkNodeCreate, user_id: str) -> BulkResult:
        succeeded: list[str] = []
        failed: list[BulkFailure] = []

        for item in payload.nodes:
            try:
                node = await self.create(item, user_id)
                succeeded.append(str(node.id))
            except Exception as exc:
                if payload.on_error == "stop":
                    raise
                failed.append(BulkFailure(node_id=f"<{item.type}:{item.title}>", error=str(exc)))

        return BulkResult(succeeded=succeeded, failed=failed)

    async def delete(self, node_id: UUID, *, cascade: bool = True) -> None:
        if not cascade:
            n = await self._repo.count_inbound_relationships(node_id)
            if n > 0:
                raise NodePreconditionError(
                    f"Node is referenced as a target by {n} relationships — delete them first or use cascade=true"
                )

        await self._repo.delete_node(node_id, cascade=cascade)

    async def bulk_delete(self, payload: BulkNodeDelete, *, cascade: bool = True) -> BulkResult:
        succeeded: list[str] = []
        failed: list[BulkFailure] = []

        for node_id in payload.node_ids:
            try:
                await self.delete(node_id, cascade=cascade)
                succeeded.append(str(node_id))
            except Exception as exc:
                if payload.on_error == "stop":
                    raise
                failed.append(BulkFailure(node_id=str(node_id), error=str(exc)))

        return BulkResult(succeeded=succeeded, failed=failed)

    async def update(self, node_id: UUID, payload: ManualNodeUpdate, user_id: str) -> KBNode:
        node = await self._repo.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)

        if node.metadata.ingestion_source != IngestionSource.MANUAL:
            raise NodePreconditionError(
                "Only manually-created nodes can be edited — use the override endpoint for pipeline nodes"
            )

        return await self._edit(node, payload, user_id)

    async def override(
        self,
        node_id: UUID,
        payload: NodeOverride,
        user_id: str,
        minimum_method: ApprovalMethod | None,
    ) -> KBNode:
        node = await self._repo.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)

        if minimum_method is not None and node.metadata.approval_method != minimum_method:
            raise NodePreconditionError("Insufficient role to override this node")

        return await self._edit(node, payload, user_id)

    async def resolve_by_ids(self, node_ids: list[UUID]) -> int:
        nodes = []
        for node_id in node_ids:
            node = await self._repo.get_node(node_id)
            if node is None:
                raise NodeNotFoundError(str(node_id))
            nodes.append(node)

        not_approved = [str(n.id) for n in nodes if n.status != NodeStatus.APPROVED]
        if not_approved:
            raise NodePreconditionError(f"Nodes not in APPROVED status: {not_approved}")

        job_id = f"manual_resolve_{uuid4()}"
        relationships = await self.resolve(nodes, job_id)
        return len(relationships)

    @track_token_budget("manual_node_resolve", uncapped=True, accumulate_to_fn=lambda self: self._usage_tracker)
    @track_latency_profile("manual_node_resolve")
    async def resolve(self, nodes: list[KBNode], job_id: str) -> list[KBRelationship]:
        """Run resolution for the given approved nodes and persist the resulting relationships."""
        result = await self._extraction_orch.run_resolution(job_id=job_id, approved=nodes)

        for rel in result.relationships:
            await self._repo.write_relationship(rel)

        return result.relationships

    @track_token_budget("manual_node_edit", uncapped=True)
    async def _edit(self, node: KBNode, payload: ManualNodeUpdate, user_id: str) -> KBNode:
        now = datetime.now(UTC)
        job_id = f"manual_{uuid4()}"
        meta_updates: dict = {
            "meeting_date": payload.meeting_date,
            "participants": payload.participants,
            "team": payload.team,
            "project": payload.project,
            "domain": payload.domain,
            "concept_fields": payload.concept_fields,
            "corrected_by": user_id,
            "corrected_at": now,
            "correction_reason": payload.reason,
        }

        updated_node = node._with(
            title=payload.title,
            description=payload.description,
            metadata=node.metadata._with(**meta_updates),
        )

        relationships = _build_relationships(node.id, payload.relationships or [], now, job_id=job_id)
        replace_rels = payload.relationships is not None
        await self._repo.update_node(
            updated_node,
            relationships=relationships or None,
            replace_outbound_rels=replace_rels,
        )

        return updated_node


def _both_current(source: KBNode, target: KBNode) -> bool:
    return source.state == NodeState.CURRENT and target.state == NodeState.CURRENT


def _build_manual_node(job_id: str, payload: ManualNodeCreate, now: datetime, user_id: str) -> KBNode:
    metadata = NodeMetadata(
        job_id=job_id,
        ingestion_source=IngestionSource.MANUAL,
        approval_method=ApprovalMethod.MANUAL,
        approved_by=user_id,
        approved_at=now,
        meeting_date=payload.meeting_date,
        participants=payload.participants,
        team=payload.team,
        project=payload.project,
        domain=payload.domain,
        concept_fields=payload.concept_fields,
    )
    return KBNode(
        id=uuid4(),
        schema_version="1.0",
        type=payload.type,
        title=payload.title,
        description=payload.description,
        confidence=1.0,
        quote_anchors=[],
        status=NodeStatus.APPROVED,
        state=NodeState.CURRENT,
        metadata=metadata,
    )


def _build_relationships(
    source_id: UUID,
    relationships: list[RelationshipInput],
    now: datetime,
    *,
    job_id: str,
) -> list[KBRelationship]:
    return [
        KBRelationship(
            source_id=source_id,
            target_id=r.target_id,
            rel_type=r.rel_type,
            job_id=job_id,
            created_at=now,
        )
        for r in relationships
    ]
