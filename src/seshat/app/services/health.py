from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from seshat.core.models.api_responses import HealthStatus

if TYPE_CHECKING:
    from seshat.app.repositories.blob_repository import BlobRepository
    from seshat.app.repositories.ops_repository import OpsRepository
    from seshat.core.config.settings import BlobStoreConfig, ObservabilityConfig


class HealthService:
    def __init__(
        self,
        ops_repo: OpsRepository,
        blob_repo: BlobRepository,
        blob_config: BlobStoreConfig,
        observability_config: ObservabilityConfig,
    ) -> None:
        self._ops = ops_repo
        self._blob = blob_repo
        self._blob_config = blob_config
        self._observability_config = observability_config

    async def check_postgres(self) -> HealthStatus:
        alive = await self._ops.is_alive()
        return HealthStatus.OK if alive else HealthStatus.ERROR

    async def check_mlflow(self) -> HealthStatus:
        url = f"{self._observability_config.mlflow_tracking_uri}/health"
        return await _check_http(url)

    async def check_blob(self) -> HealthStatus:
        if self._blob_config.endpoint_url:
            return await _check_http(f"{self._blob_config.endpoint_url}/_localstack/health")

        try:
            await self._blob._store.client.head_bucket(Bucket=self._blob_config.bucket)
            return HealthStatus.OK
        except Exception:
            return HealthStatus.ERROR


async def _check_http(url: str) -> HealthStatus:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        return HealthStatus.OK
    except httpx.HTTPError:
        return HealthStatus.ERROR
