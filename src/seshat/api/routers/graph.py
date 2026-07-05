from __future__ import annotations

from typing import Annotated
from uuid import UUID  # noqa: TC003  — FastAPI resolves path-param annotations at runtime

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
    ImpactResponse,
    NodeDetailResponse,
    NodeListResponse,
    NodeSearchResponse,
)
from seshat.models.enums import ApprovalMethod, RelationshipType, SearchMode, UserRole
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
    nodes = await state.graph_service.query(node_filter)
    return NodeListResponse(nodes=nodes)


@router.get(
    "/search",
    summary="Hybrid semantic search over KB nodes",
    responses={
        200: {"description": "Matching nodes with neighbours, ordered by relevance"},
        401: {"description": "Missing or invalid API key"},
    },
)
async def search_graph(
    state: Annotated[AppState, Depends(get_app_state)],
    q: str,
    node_filter: Annotated[NodeFilter, Depends()],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    search_mode: SearchMode = SearchMode.SEMANTIC,
) -> NodeSearchResponse:
    results = await state.graph_service.search(query=q, limit=limit, node_filter=node_filter, mode=search_mode)
    return NodeSearchResponse(results=results)


@router.get(
    "/{node_id}",
    summary="Fetch a single KB node by ID",
    responses={
        200: {"description": "The node"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Node not found"},
    },
)
async def get_node(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
) -> KBNode:
    try:
        return await state.graph_service.get_node(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")


@router.get(
    "/{node_id}/neighbours",
    summary="List depth-1 neighbours of a node (both directions, active only)",
    responses={
        200: {"description": "Directly connected active nodes"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Node not found"},
    },
)
async def get_node_neighbours(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
) -> list[KBNode]:
    try:
        return await state.graph_service.get_node_neighbours(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")


@router.get(
    "/{node_id}/detail",
    summary="Fetch a node together with its depth-1 neighbours",
    responses={
        200: {"description": "Node and directly connected active nodes"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Node not found"},
    },
)
async def get_node_detail(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
) -> NodeDetailResponse:
    try:
        return await state.graph_service.get_node_detail(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")


@router.get(
    "/{node_id}/impact",
    summary="Multi-hop inbound traversal — nodes that upstream-influence this one",
    responses={
        200: {"description": "Upstream nodes with their traversal depth"},
        401: {"description": "Missing or invalid API key"},
        422: {"description": "depth out of allowed range [1, 3]"},
    },
)
async def impact_traversal(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
    rel_types: str | None = None,
    min_confidence: float = 0.0,
) -> ImpactResponse:
    rel_type_filter = [RelationshipType(r.strip()) for r in rel_types.split(",")] if rel_types else None
    return await state.graph_service.traverse_impact(node_id, depth, rel_type_filter, min_confidence)


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
    return await state.graph_service.bulk_create(payload, user.user_id)


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
    try:
        relationships = await state.graph_service.resolve_by_ids(payload.node_ids)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return ResolveResponse(relationships_created=relationships)


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
    return await state.graph_service.create(payload, user.user_id)


@router.put(
    "/{node_id}",
    summary="Alter a manually-created node",
    responses={
        200: {"description": "Updated node"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Node not found"},
        409: {"description": "Node not eligible for update (e.g. not manually created)"},
    },
)
async def update_node(
    node_id: UUID,
    payload: ManualNodeUpdate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    try:
        return await state.graph_service.update(node_id, payload, user.user_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.put(
    "/{node_id}/override",
    summary="Alter any node with correction metadata, role-gated",
    responses={
        200: {"description": "New node version created"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Node not found"},
        409: {"description": "Node not eligible for override"},
    },
)
async def override_node(
    node_id: UUID,
    payload: NodeOverride,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    minimum_method = None if user.role.is_at_least(UserRole.ADMIN) else ApprovalMethod.AUTO
    try:
        return await state.graph_service.override(node_id, payload, user.user_id, minimum_method=minimum_method)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


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
    return await state.graph_service.bulk_delete(payload, cascade=cascade)


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
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
    cascade: bool = True,
) -> None:
    try:
        await state.graph_service.delete(node_id, cascade=cascade)
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
