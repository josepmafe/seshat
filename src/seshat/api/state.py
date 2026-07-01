from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.ops.ledger import OpsLedger
    from seshat.vector_store.base_store import AbstractVectorStore
    from seshat.worker.bootstrap import WorkerContext
    from seshat.worker.manual_ingestion import ManualIngestionService
    from seshat.worker.pipeline_runner import PipelineRunner
    from seshat.worker.queue import AsyncioTaskQueue


@dataclass
class AppState:
    config: SeshatConfig
    kb_store: PostgresKBStore
    vector_store: AbstractVectorStore
    manual_ingestion: ManualIngestionService
    ops: OpsLedger
    queue: AsyncioTaskQueue
    runner: PipelineRunner
    blob_store: S3BlobStore

    @classmethod
    def from_context(
        cls,
        ctx: WorkerContext,
        config: SeshatConfig,
        runner: PipelineRunner,
        queue: AsyncioTaskQueue,
    ) -> AppState:
        return cls(
            config=config,
            kb_store=ctx.kb_store,
            vector_store=ctx.vector_store,
            manual_ingestion=ctx.manual_ingestion,
            ops=ctx.ops,
            queue=queue,
            runner=runner,
            blob_store=ctx.blob_store,
        )
