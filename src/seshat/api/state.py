from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seshat.blob_store.factory import get_blob_store
from seshat.knowledge_store.factory import get_kb_store
from seshat.ops_store.factory import get_ops_store
from seshat.pipeline.bootstrap import build_extraction_orchestrator, build_ingestion_orchestrator, build_vector_store
from seshat.repositories.blob_repository import BlobRepository
from seshat.repositories.node_repository import NodeRepository
from seshat.repositories.ops_repository import OpsRepository
from seshat.services.admin_service import AdminService
from seshat.services.graph_service import GraphService
from seshat.services.health_service import HealthService
from seshat.services.job_service import JobService
from seshat.worker.queue import AsyncioTaskQueue

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.config.settings import SeshatConfig


@dataclass
class AppState:
    config: SeshatConfig
    admin_service: AdminService
    health_service: HealthService
    graph_service: GraphService
    job_service: JobService


@asynccontextmanager
async def build_app_state(config: SeshatConfig) -> AsyncIterator[AppState]:
    ops_store = get_ops_store(config)
    await ops_store.connect()

    kb_store = get_kb_store(config)
    await kb_store.connect()

    blob_store = get_blob_store(config)
    await blob_store.connect()

    try:
        vector_store = build_vector_store(config)
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
        graph_service = GraphService(node_repo, extraction_orchestrator)
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
        await kb_store.close()
        await blob_store.close()
        await ops_store.close()
