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
from seshat.worker.manual_ingestion import ManualIngestionService
from seshat.worker.writing_stage import WritingStage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.pipeline.ingestion.orchestrator import IngestionOrchestrator
    from seshat.vector_store.base_store import AbstractVectorStore


@dataclass
class WorkerContext:
    ingestion_orchestrator: IngestionOrchestrator
    extraction_orchestrator: ExtractionOrchestrator
    writing_stage: WritingStage
    ops_repo: OpsRepository
    kb_store: PostgresKBStore
    vector_store: AbstractVectorStore
    manual_ingestion: ManualIngestionService
    blob_store: S3BlobStore
    node_repo: NodeRepository
    blob_repo: BlobRepository


@asynccontextmanager
async def build_worker_context(seshat_config: SeshatConfig) -> AsyncIterator[WorkerContext]:
    ops_store = await get_ops_store(seshat_config)

    kb_store = get_kb_store(seshat_config)
    await kb_store.connect()

    blob_store = get_blob_store(seshat_config)
    await blob_store.connect()

    try:
        vector_store = build_vector_store(seshat_config)
        node_repo = NodeRepository(kb_store, vector_store)
        blob_repo = BlobRepository(blob_store)
        ingestion_orchestrator = build_ingestion_orchestrator(seshat_config, blob_repo)
        extraction_orchestrator = build_extraction_orchestrator(seshat_config, node_repo, blob_repo)
        writing_stage = WritingStage(kb_store, vector_store)
        manual_ingestion = ManualIngestionService(kb_store, vector_store, extraction_orchestrator)
        yield WorkerContext(
            ingestion_orchestrator=ingestion_orchestrator,
            extraction_orchestrator=extraction_orchestrator,
            writing_stage=writing_stage,
            manual_ingestion=manual_ingestion,
            node_repo=node_repo,
            blob_repo=blob_repo,
            ops_repo=OpsRepository(ops_store),
            kb_store=kb_store,
            vector_store=vector_store,
            blob_store=blob_store,
        )
    finally:
        await kb_store.close()
        await blob_store.close()
        await ops_store.close()
