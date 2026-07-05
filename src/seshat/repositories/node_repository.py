from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.models.enums import GraphDirection, NodeState, NodeStatus, RelationshipType, SearchMode
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.knowledge_store.pg_store import PostgresKBStore, _Conn
    from seshat.models.api_graph import NodeFilter, SearchResult
    from seshat.models.nodes import ExtractionResult, KBNode, KBRelationship
    from seshat.vector_store.base_store import AbstractVectorStore

logger = get_logger(__name__)

_STATE_TRANSITIONS = {
    RelationshipType.SUPERSEDES: NodeState.SUPERSEDED,
    RelationshipType.AMENDS: NodeState.AMENDED,
}


class NodeRepository:
    """Single persistence façade for all node operations.

    Coordinates PostgresKBStore and AbstractVectorStore — callers never touch
    the raw stores directly. The two-phase commit pattern (KB transaction →
    VS operation) is handled internally.

    Public methods accept UUID for node identifiers; string conversion to the
    store layer happens here at the boundary.
    """

    def __init__(self, kb_store: PostgresKBStore, vector_store: AbstractVectorStore) -> None:
        self._kb = kb_store
        self._vs = vector_store

    # -- Write (KB + VS) -------------------------------------------------------

    async def write_node(self, node: KBNode, *, relationships: list[KBRelationship] | None = None) -> None:
        async with self._kb.transaction() as conn:
            await self._kb.write_node(node, conn=conn)
            if relationships:
                for rel in relationships:
                    await self._kb.write_relationship(rel, conn=conn)

        await self._vs.upsert(str(node.id), node.vector_store_text, node.metadata.model_dump(mode="json"))

    async def update_node(
        self,
        node: KBNode,
        *,
        relationships: list[KBRelationship] | None = None,
        replace_outbound_rels: bool = False,
    ) -> None:
        async with self._kb.transaction() as conn:
            await self._kb.update_node(node, conn=conn)
            if replace_outbound_rels:
                await self._kb.delete_relationships_for_node(str(node.id), cascade=False, conn=conn)
            if relationships:
                for rel in relationships:
                    await self._kb.write_relationship(rel, conn=conn)

        await self._vs.upsert(str(node.id), node.vector_store_text, node.metadata.model_dump(mode="json"))

    async def delete_node(self, node_id: UUID, *, cascade: bool = True) -> None:
        # Before deleting, find nodes whose state was set by an outbound SUPERSEDES/AMENDS
        # relationship from this node. If no other source still supersedes/amends them after
        # this deletion, revert them to CURRENT.
        # Alternative: introduce NodeState.ORPHANED instead of reverting, to signal that a
        # human should review the node's status — deferred until there is evidence of need.
        sid = str(node_id)
        targets = await self._kb.get_outbound_state_transition_targets(sid)

        async with self._kb.transaction() as conn:
            for target_id in targets:
                remaining = await self._kb.count_remaining_state_transition_sources(
                    target_id, excluding_source_id=sid, conn=conn
                )
                if remaining == 0:
                    await self._kb.update_node_state(target_id, NodeState.CURRENT, conn=conn)
                    logger.info("Reverted node %s to CURRENT (source %s deleted)", target_id, sid)

            await self._kb.delete_relationships_for_node(sid, cascade=cascade, conn=conn)
            await self._kb.delete_node(sid, conn=conn)

        await self._vs.delete(sid)

    # -- Write (KB only) -------------------------------------------------------

    async def write_relationship(self, rel: KBRelationship) -> None:
        await self._kb.write_relationship(rel)

    async def update_node_state(self, node_id: UUID, state: NodeState) -> None:
        await self._kb.update_node_state(str(node_id), state)

    # -- Batch -----------------------------------------------------------------

    async def write_batch(self, result: ExtractionResult) -> int:
        """Write an extraction result to KB and VS.

        Applies state transitions to superseded/amended existing nodes, writes
        approved nodes and their relationships within a KB transaction, then
        upserts VS embeddings outside the transaction. Returns the count of
        written nodes.
        """
        async with self._kb.transaction() as conn:
            await self._apply_state_transitions(result.relationships, conn)
            approved_ids = await self._write_nodes_and_relationships(result.nodes, result.relationships, conn)

        await self._upsert_vectors(result.nodes, approved_ids)

        return len(approved_ids)

    async def _apply_state_transitions(self, relationships: list[KBRelationship], conn: _Conn) -> None:
        for rel in relationships:
            new_state = _STATE_TRANSITIONS.get(rel.rel_type)
            if not new_state:
                continue

            try:
                await self._kb.update_node_state(str(rel.target_id), new_state, conn=conn)
                logger.info(
                    "Node %s state → %s (due to %s from %s)",
                    rel.target_id,
                    new_state,
                    rel.rel_type,
                    rel.source_id,
                )
            except KeyError:
                logger.warning(
                    "update_node_state: target node %s not found (rel_type=%s, source=%s) — skipping",
                    rel.target_id,
                    rel.rel_type,
                    rel.source_id,
                )

    async def _write_nodes_and_relationships(
        self, nodes: list[KBNode], relationships: list[KBRelationship], conn: _Conn
    ) -> set[UUID]:
        all_ids: set[UUID] = {n.id for n in nodes}
        approved_ids: set[UUID] = set()

        for node in nodes:
            if node.status != NodeStatus.APPROVED:
                continue
            await self._kb.write_node(node, conn=conn)
            approved_ids.add(node.id)

        for rel in relationships:
            target_ok = rel.target_id in approved_ids or rel.target_id not in all_ids
            if rel.source_id in approved_ids and target_ok:
                await self._kb.write_relationship(rel, conn=conn)

        return approved_ids

    async def _upsert_vectors(self, nodes: list[KBNode], approved_ids: set[UUID]) -> None:
        for node in nodes:
            if node.id not in approved_ids:
                continue
            await self._vs.upsert(str(node.id), node.vector_store_text, node.metadata.model_dump(mode="json"))

    # -- Read (delegates to KB) ------------------------------------------------

    async def get_node(self, node_id: UUID) -> KBNode | None:
        return await self._kb.get_node(str(node_id))

    async def query(self, node_filter: NodeFilter) -> list[KBNode]:
        return await self._kb.query(node_filter)

    async def paginated_query(self, node_filter: NodeFilter) -> list[KBNode]:
        return await self._kb.paginated_query(node_filter)

    async def get_neighbours(
        self,
        node_id: UUID,
        rel_types: list[RelationshipType] | None = None,
        direction: GraphDirection = GraphDirection.BOTH,
    ) -> list[KBNode]:
        return await self._kb.get_neighbours(str(node_id), rel_types=rel_types, direction=direction)

    async def get_node_relationships(self, node_id: UUID) -> list[KBRelationship]:
        return await self._kb.get_node_relationships(str(node_id))

    async def count_inbound_relationships(self, node_id: UUID) -> int:
        return await self._kb.count_inbound_relationships(str(node_id))

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
        score_threshold: float | None = None,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> list[SearchResult]:
        return await self._vs.search(
            query,
            top_k=top_k,
            node_filter=node_filter,
            exclude_job_id=exclude_job_id,
            score_threshold=score_threshold,
            mode=mode,
        )
