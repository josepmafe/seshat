from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.models.nodes import ExtractionResult
    from seshat.ops.ledger import OpsLedger
    from seshat.worker.pipeline_runner import PipelineRunner
    from seshat.worker.queue import AsyncioTaskQueue


@dataclass
class AppState:
    ops: OpsLedger
    kb_store: PostgresKBStore
    config: SeshatConfig
    queue: AsyncioTaskQueue
    results: dict[str, ExtractionResult]
    runner: PipelineRunner
