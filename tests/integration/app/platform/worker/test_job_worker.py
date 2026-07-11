from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import mlflow
import pytest
import yaml

from seshat.app.pipeline.ingestion.orchestrator import IngestionOrchestrator
from seshat.app.platform.worker.queue import AsyncioTaskQueue
from seshat.app.repositories.node_repository import NodeRepository
from seshat.app.repositories.ops_repository import OpsRepository
from seshat.app.services.job import JobService
from seshat.core.config.settings import KBStoreConfig, OpsStoreConfig, TranscriptionConfig
from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.enums import JobStatus, NodeStatus
from seshat.core.models.nodes import IdentificationResult, ResolutionResult
from seshat.core.models.submission import JobSubmissionRequest
from seshat.core.models.transcript import TranscriptMetadata
from seshat.infra.knowledge_store.pg_store import PostgresKBStore
from seshat.infra.ops_store.pg_store import PostgresOpsStore
from tests.helpers import make_node
from tests.integration.conftest import SKIP_IF_NO_LOCALSTACK, SKIP_IF_NO_POSTGRES

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES, SKIP_IF_NO_LOCALSTACK]

_MEETING_DATE = date(2026, 1, 15)
_USER_ID = "worker-test-user"
_RAW_YAML = yaml.dump(
    {"date": _MEETING_DATE.isoformat(), "content": "We decided to use PostgreSQL for the user database."}
).encode()


@pytest.fixture
async def ops_repo(pg_test_url):
    pool = await asyncpg.create_pool(pg_test_url)
    store = PostgresOpsStore(OpsStoreConfig(), pg_test_url)
    store._pool = pool
    yield OpsRepository(store)
    await pool.execute("TRUNCATE ops.jobs CASCADE")
    await pool.close()


@pytest.fixture
async def kb_store(pg_test_url):
    store = PostgresKBStore(KBStoreConfig(), pg_test_url)
    await store.connect()
    yield store
    await store.pool.execute(f"TRUNCATE {store._schema}.kb_relationships, {store._schema}.kb_nodes CASCADE")
    await store.close()


@pytest.fixture
def fake_vector_store():
    vs = MagicMock()
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()
    vs.search = AsyncMock(return_value=[])
    return vs


@pytest.fixture
def fake_extraction_orch():
    node = make_node("c3a-worker-node", status=NodeStatus.PENDING_REVIEW)

    async def _canned_identification(doc, job_id, *, user_id=None, config_override=None):
        return IdentificationResult(job_id=job_id, nodes=[node], confidence_breakdowns={})

    async def _canned_resolution(job_id, *, approved=None):
        return ResolutionResult(job_id=job_id, relationships=[])

    orch = MagicMock()
    orch.run_identification = AsyncMock(side_effect=_canned_identification)
    orch.run_resolution = AsyncMock(side_effect=_canned_resolution)
    orch._config = MagicMock(auto_mode=False)
    return orch


@pytest.fixture
def job_service_and_queue(blob_store, ops_repo, kb_store, fake_vector_store, fake_extraction_orch, tmp_path):
    mlflow.set_tracking_uri("sqlite:///" + str(tmp_path / "mlflow.db"))

    ingestion = IngestionOrchestrator(MagicMock(), blob_store, TranscriptionConfig())
    node_repo = NodeRepository(kb_store, fake_vector_store)
    queue = AsyncioTaskQueue()

    config = MagicMock()
    config.api.max_jobs_per_user_per_hour = 100
    config.api.max_concurrent_jobs = 100

    svc = JobService(config, ops_repo, blob_store, node_repo, fake_extraction_orch, ingestion, queue)
    return svc, queue


class TestJobWorkerE2E:
    async def test_text_job_completes_to_done(self, job_service_and_queue, ops_repo, kb_store, blob_store):
        svc, queue = job_service_and_queue

        submission = JobSubmissionRequest(
            source_type="text",
            metadata=TranscriptMetadata(meeting_date=_MEETING_DATE),
            auto_mode=True,
        )
        response = await svc.submit(_RAW_YAML, "meeting.yaml", submission, _USER_ID)
        job_id = response.job_id

        # Drain: _run_pre_approval auto-enqueues _run_post_approval when no pending nodes
        while True:
            pending = [t for t in queue._tasks.values() if not t.done()]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)

        row = await ops_repo.get_job(job_id)
        assert row is not None
        assert row["status"] == JobStatus.DONE

        blob = await blob_store.get_curated_extraction(_MEETING_DATE, job_id)
        assert blob is not None

        nodes = await kb_store.query(NodeFilter())
        assert len(nodes) >= 1
