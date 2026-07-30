from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import mlflow
from mlflow import MlflowClient

from seshat.app.pipeline.bootstrap import (
    build_extraction_orchestrator,
    build_ingestion_orchestrator,
    get_search_engine,
)
from seshat.app.platform.worker.queue import AsyncioTaskQueue
from seshat.app.repositories.blob_repository import BlobRepository
from seshat.app.repositories.node_repository import NodeRepository
from seshat.app.repositories.ops_repository import OpsRepository
from seshat.app.services.admin import AdminService
from seshat.app.services.graph import GraphService
from seshat.app.services.health import HealthService
from seshat.app.services.job import JobService
from seshat.infra.blob_store.factory import get_blob_store
from seshat.infra.knowledge_store.factory import get_kb_store
from seshat.infra.ops_store.factory import get_ops_store
from seshat.infra.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from seshat.core.config.settings import ObservabilityConfig, SeshatConfig


@dataclass
class AppState:
    config: SeshatConfig
    admin_service: AdminService
    health_service: HealthService
    graph_service: GraphService
    job_service: JobService


@asynccontextmanager
async def build_app_state(config: SeshatConfig) -> AsyncGenerator[AppState]:
    ops_store = get_ops_store(config)
    await ops_store.connect()

    kb_store = get_kb_store(config)
    await kb_store.connect()

    blob_store = get_blob_store(config)
    await blob_store.connect()

    mlflow_client = MlflowClient()
    search_run_id = _get_search_run_id(mlflow_client, config.observability)

    try:
        vector_store = get_vector_store(config)
        node_repo = NodeRepository(kb_store, vector_store)
        blob_repo = BlobRepository(blob_store)
        extraction_orchestrator = build_extraction_orchestrator(config, node_repo, blob_repo)
        ingestion_orchestrator = build_ingestion_orchestrator(config, blob_repo)
        ops_repo = OpsRepository(ops_store)
        admin_service = AdminService(ops_repo=ops_repo)
        health_service = HealthService(
            ops_repo=ops_repo,
            blob_repo=blob_repo,
            blob_config=config.blob_store,
            observability_config=config.observability,
        )
        graph_search_engine = get_search_engine(config, vector_store, disable_multi_query=True)
        graph_service = GraphService(node_repo, extraction_orchestrator, graph_search_engine, search_run_id)
        queue = AsyncioTaskQueue()
        job_service = JobService(
            config,
            ops_repo,
            blob_repo,
            node_repo,
            extraction_orchestrator,
            ingestion_orchestrator,
            queue,
        )
        yield AppState(
            config=config,
            admin_service=admin_service,
            health_service=health_service,
            graph_service=graph_service,
            job_service=job_service,
        )
    finally:
        mlflow_client.set_terminated(search_run_id)
        await kb_store.close()
        await blob_store.close()
        await ops_store.close()


def _get_search_run_id(mlflow_client: MlflowClient, config: ObservabilityConfig):
    """Create one long-lived MLflow run for all GraphService.search() calls.

    Created via MlflowClient so it never touches mlflow's thread-local active-run stack,
    unlike `mlflow.start_run()`. Note that `setup_mlflow()` has already run by this point
    (API startup, before build_app_state), so the experiment exists.
    """
    experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
    assert experiment is not None
    run_name = f"search-run-{datetime.now(tz=UTC).isoformat(timespec='minutes')}"
    search_run = mlflow_client.create_run(experiment.experiment_id, tags={"source": "graph_search"}, run_name=run_name)
    return search_run.info.run_id
