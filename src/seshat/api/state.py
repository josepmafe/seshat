from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.repositories.ops_repository import OpsRepository
    from seshat.services.graph_service import GraphService
    from seshat.services.job_service import JobService
    from seshat.vector_store.base_store import AbstractVectorStore


@dataclass
class AppState:
    config: SeshatConfig
    kb_store: PostgresKBStore
    vector_store: AbstractVectorStore
    manual_ingestion: GraphService
    ops: OpsRepository
    job_service: JobService
    blob_store: S3BlobStore
