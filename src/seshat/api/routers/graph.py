from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from seshat.api.dependencies import CurrentUser, get_app_state, require_role
from seshat.api.state import AppState
from seshat.models.api_graph import (
    BulkNodeCreate,
    BulkNodeDelete,
    BulkResult,
    ManualNodeCreate,
    ManualNodeUpdate,
    NodeFilter,
    NodeOverride,
    ResolveRequest,
    ResolveResponse,
)
from seshat.models.api_responses import (
    ImpactNode,
    ImpactResponse,
    NodeDetailResponse,
    NodeListResponse,
)
from seshat.models.enums import (
    ApprovalMethod,
    GraphDirection,
    NodeState,
    NodeStatus,
    RelationshipType,
    UserRole,
)
from seshat.models.nodes import KBNode
from seshat.services.graph_service import NodeNotFoundError, NodePreconditionError

router = APIRouter(prefix="/graph", tags=["graph"], dependencies=[Depends(require_role(UserRole.VIEWER))])


@router.get(
    "",
    summary="Query knowledge graph nodes",
    responses={
        200: {"description": "List of matching nodes (may be empty)"},
        401: {"description": "Missing or invalid API key"},
    },
)
async def query_graph(
    state: Annotated[AppState, Depends(get_app_state)],
    node_filter: Annotated[NodeFilter, Depends()],
) -> NodeListResponse:
    nodes = await state.kb_store.query(node_filter)
    return NodeListResponse(nodes=nodes)


@router.get(
    "/{node_id}",
    summary="Get a single node with its neighbours",
    responses={
        200: {"description": "Node and active neighbours"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Node not found"},
    },
)
async def get_node(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> NodeDetailResponse:
    node = await state.kb_store.get_node(node_id)
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")

    neighbours = await state.kb_store.get_neighbours(node_id, direction=GraphDirection.BOTH)
    active_neighbours = [n for n in neighbours if _both_current(node, n)]
    return NodeDetailResponse(node=node, neighbours=active_neighbours)


@router.get(
    "/{node_id}/impact",
    summary="Traverse inbound impact graph from a node",
    responses={
        200: {"description": "Nodes with traversal depth"},
        401: {"description": "Missing or invalid API key"},
        422: {"description": "depth out of allowed range [1, 3]"},
    },
)
async def impact_traversal(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
    rel_types: str | None = None,
    min_confidence: float = 0.0,
) -> ImpactResponse:
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

    impact_nodes: list[ImpactNode] = []
    for nid, hop in visited.items():
        n = await state.kb_store.get_node(nid)
        if n:
            impact_nodes.append(ImpactNode(node=n, traversal_depth=hop))

    return ImpactResponse(nodes=impact_nodes)


@router.post(
    "/bulk",
    summary="Bulk create nodes",
    responses={
        200: {"description": "Succeeded and failed node IDs"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
    },
)
async def bulk_create_nodes(
    payload: BulkNodeCreate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> BulkResult:
    result = await state.manual_ingestion.bulk_create(payload, user.user_id)
    return result


@router.post(
    "/nodes/resolve",
    summary="Trigger resolution for a set of approved nodes",
    responses={
        200: {"description": "Number of relationships created"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "One or more node IDs not found"},
        422: {"description": "One or more nodes not in APPROVED status"},
    },
)
async def resolve_nodes(
    payload: ResolveRequest,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> ResolveResponse:
    nodes = []
    for node_id in payload.node_ids:
        node = await state.kb_store.get_node(str(node_id))
        if node is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node {node_id} not found")
        nodes.append(node)

    not_approved = [str(n.id) for n in nodes if n.status != NodeStatus.APPROVED]
    if not_approved:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Nodes not in APPROVED status: {not_approved}",
        )

    job_id = f"manual_resolve_{uuid4()}"
    relationships = await state.manual_ingestion.resolve(nodes, job_id)
    return ResolveResponse(relationships_created=len(relationships))


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Manually create a node",
    responses={
        201: {"description": "Node created"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
    },
)
async def create_node(
    payload: ManualNodeCreate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    node = await state.manual_ingestion.create(payload, user.user_id)
    return node


@router.put(
    "/{node_id}",
    summary="Update a manually-created node",
    responses={
        200: {"description": "Updated node"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Node not found"},
        409: {"description": "Node not eligible for update (e.g. not manually created)"},
    },
)
async def update_node(
    node_id: str,
    payload: ManualNodeUpdate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    try:
        node = await state.manual_ingestion.update(node_id, payload, user.user_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    return node


@router.put(
    "/{node_id}/override",
    summary="Override a node (creates a SUPERSEDES/AMENDS successor)",
    responses={
        200: {"description": "New node version created"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Node not found"},
        409: {"description": "Node not eligible for override"},
    },
)
async def override_node(
    node_id: str,
    payload: NodeOverride,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    minimum_method = None if user.role.is_at_least(UserRole.ADMIN) else ApprovalMethod.AUTO
    try:
        node = await state.manual_ingestion.override(node_id, payload, user.user_id, minimum_method=minimum_method)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    return node


@router.delete(
    "/bulk",
    summary="Bulk delete nodes",
    responses={
        200: {"description": "Succeeded and failed node IDs"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Admin role required"},
    },
)
async def bulk_delete_nodes(
    payload: BulkNodeDelete,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
    cascade: bool = True,
) -> BulkResult:
    result = await state.manual_ingestion.bulk_delete(payload, cascade=cascade)
    return result


@router.delete(
    "/{node_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a node",
    responses={
        204: {"description": "Node deleted"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Admin role required"},
        409: {"description": "Cannot delete node with inbound relationships (use cascade=true)"},
    },
)
async def delete_node(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
    cascade: bool = True,
) -> None:
    try:
        await state.manual_ingestion.delete(node_id, cascade=cascade)
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


def _both_current(source: KBNode, target: KBNode) -> bool:
    return source.state == NodeState.CURRENT and target.state == NodeState.CURRENT
