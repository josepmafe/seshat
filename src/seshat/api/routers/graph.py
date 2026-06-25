from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from seshat.api.dependencies import get_app_state, require_role
from seshat.api.state import AppState
from seshat.models.api import NodeFilter
from seshat.models.enums import ConceptType, GraphDirection, IngestionSource, NodeState, RelationshipType, UserRole

if TYPE_CHECKING:
    from seshat.models.nodes import KBNode


router = APIRouter(prefix="/graph", tags=["graph"], dependencies=[Depends(require_role(UserRole.VIEWER))])


@router.get("")
async def query_graph(
    state: Annotated[AppState, Depends(get_app_state)],
    node_type: ConceptType | None = None,
    team: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    ingestion_source: IngestionSource | None = None,
    min_confidence: float | None = None,
    node_state: NodeState | None = None,
):
    node_filter = NodeFilter(
        node_type=node_type,
        team=team,
        project=project,
        domain=domain,
        ingestion_source=ingestion_source,
        min_confidence=min_confidence,
        state=node_state,
    )
    nodes = await state.kb_store.query(node_filter)
    return {"nodes": [n.model_dump() for n in nodes]}


@router.get("/{node_id}")
async def get_node(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
):
    node = await state.kb_store.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    neighbours = await state.kb_store.get_neighbours(node_id, direction=GraphDirection.BOTH)
    active_neighbours = [n for n in neighbours if _both_current(node, n)]
    return {"node": node.model_dump(), "neighbours": [n.model_dump() for n in active_neighbours]}


@router.get("/{node_id}/impact")
async def impact_traversal(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
    rel_types: str | None = None,
    min_confidence: float = 0.0,
):
    rel_type_filter = [RelationshipType(r.strip()) for r in rel_types.split(",")] if rel_types else None

    visited: dict[str, int] = {}
    frontier = [node_id]
    for hop in range(1, depth + 1):
        next_frontier = []
        for nid in frontier:
            neighbours = await state.kb_store.get_neighbours(
                nid, rel_types=rel_type_filter, direction=GraphDirection.INBOUND
            )
            for n in neighbours:
                if str(n.id) not in visited and n.confidence >= min_confidence:
                    visited[str(n.id)] = hop
                    next_frontier.append(str(n.id))

        frontier = next_frontier

    nodes = []
    for nid, hop in visited.items():
        n = await state.kb_store.get_node(nid)
        if n:
            nodes.append({**n.model_dump(), "traversal_depth": hop})

    return {"nodes": nodes}


def _both_current(source: KBNode, target: KBNode) -> bool:
    return source.state == NodeState.CURRENT and target.state == NodeState.CURRENT
