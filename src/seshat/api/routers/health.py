from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from seshat.api.dependencies import get_app_state
from seshat.api.state import AppState
from seshat.core.models.api_responses import HealthResponse, HealthStatus

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    summary="API health check",
    response_model=HealthResponse,
    responses={200: {"description": "API is healthy"}, 503: {"description": "API is degraded"}},
)
async def health() -> HealthResponse:
    return HealthResponse(status=HealthStatus.OK)


@router.get(
    "/components",
    response_model=HealthResponse,
    summary="Service health check",
    responses={200: {"description": "All components healthy"}, 503: {"description": "One or more components degraded"}},
)
async def components_health(state: Annotated[AppState, Depends(get_app_state)], response: Response) -> HealthResponse:
    svc = state.health_service
    components = {
        "postgres": await svc.check_postgres(),
        "mlflow": await svc.check_mlflow(),
        "blob_store": await svc.check_blob(),
    }
    overall = HealthStatus.OK if all(v == HealthStatus.OK for v in components.values()) else HealthStatus.DEGRADED

    if overall != HealthStatus.OK:
        response.status_code = 503

    return HealthResponse(status=overall, components=components)
