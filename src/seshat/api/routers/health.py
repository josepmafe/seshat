from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from seshat.api.dependencies import get_app_state
from seshat.api.state import AppState


class HealthStatus(StrEnum):
    OK = auto()
    DEGRADED = auto()
    ERROR = auto()


class HealthResponse(BaseModel):
    status: HealthStatus
    components: dict[str, HealthStatus]


router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    summary="API health check",
    response_model=HealthResponse,
    responses={200: {"description": "API is healthy"}, 503: {"description": "API is degraded"}},
)
async def health() -> HealthResponse:
    return HealthResponse(status=HealthStatus.OK, components={})


@router.get(
    "/components",
    response_model=HealthResponse,
    summary="Service health check",
    responses={200: {"description": "All components healthy"}, 503: {"description": "One or more components degraded"}},
)
async def components_health(state: Annotated[AppState, Depends(get_app_state)], response: Response) -> HealthResponse:
    config = state.config

    postgres = await _check_postgres(state)
    mlflow = await _check_http(f"{config.observability.mlflow_tracking_uri}/health")
    blob = await _check_blob_store(state)

    components = {"postgres": postgres, "mlflow": mlflow, "blob_store": blob}
    overall = HealthStatus.OK if all(v == HealthStatus.OK for v in components.values()) else HealthStatus.DEGRADED

    if overall != HealthStatus.OK:
        response.status_code = 503

    return HealthResponse(status=overall, components=components)


async def _check_postgres(state: AppState) -> HealthStatus:
    try:
        await state.ops._pool.fetchval("SELECT 1")
        return HealthStatus.OK
    except Exception:
        return HealthStatus.ERROR


async def _check_blob_store(state: AppState) -> HealthStatus:
    config = state.config.blob_store
    if config.endpoint_url:
        return await _check_http(f"{config.endpoint_url}/_localstack/health")

    try:
        await state.blob_store.client.head_bucket(Bucket=config.bucket)
        return HealthStatus.OK
    except Exception:
        return HealthStatus.ERROR


async def _check_http(url: str) -> HealthStatus:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(url)
        return HealthStatus.OK
    except httpx.HTTPError:
        return HealthStatus.ERROR
