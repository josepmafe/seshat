from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import asyncpg
import pytest

from seshat.app.repositories.ops_repository import OpsRepository
from seshat.core.config.settings import OpsStoreConfig
from seshat.core.models.enums import JobStatus
from seshat.infra.ops_store.pg_store import PostgresOpsStore
from tests.integration.conftest import SKIP_IF_NO_POSTGRES

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]


@pytest.fixture
async def repo(pg_test_url: str) -> AsyncGenerator[OpsRepository]:
    pool = await asyncpg.create_pool(pg_test_url)
    store = PostgresOpsStore(OpsStoreConfig(schema_name="ops"), pg_test_url)
    store._pool = pool
    yield OpsRepository(store)
    await pool.execute("TRUNCATE ops.jobs CASCADE")
    await pool.close()


class TestCreateJob:
    async def test_row_contains_all_columns(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "audio", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-1/raw/input.mp3"

        await repo.create_job("job-1", "user-1", "audio", None, datetime.now(UTC), meeting_date, submission, raw_key)

        row = await repo.get_job("job-1")
        assert row is not None
        assert str(row["meeting_date"]) == "2026-06-01"
        assert json.loads(row["submission"])["source_type"] == "audio"
        assert row["raw_blob_key"] == raw_key
        assert row["status"] == "pending"

    async def test_not_null_constraints_are_satisfied(self, repo: OpsRepository):
        pool = repo._store._pool
        with pytest.raises(asyncpg.NotNullViolationError):
            await pool.execute(
                "INSERT INTO ops.jobs (job_id, user_id, status, source_type, created_at, updated_at)"
                " VALUES ($1, $2, 'pending', $3, $4, $4)",
                "job-bad",
                "user-1",
                "audio",
                datetime.now(UTC),
            )


class TestUpdateJobStatus:
    async def test_status_transitions(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-2/raw/input.txt"

        await repo.create_job("job-2", "user-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key)
        await repo.update_job_status("job-2", JobStatus.IDENTIFYING)

        row = await repo.get_job("job-2")
        assert row is not None
        assert row["status"] == "identifying"


class TestFailJob:
    async def test_fail_job_sets_status_and_error(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-3/raw/input.txt"

        await repo.create_job("job-3", "user-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key)
        await repo.fail_job("job-3", "pipeline", "something broke", recoverable=True)

        row = await repo.get_job("job-3")
        assert row is not None
        assert row["status"] == "failed"
        assert row["error_payload"] is not None


class TestContentHashDedup:
    async def test_first_done_job_found_by_content_hash(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-hash-1/raw/input.txt"

        await repo.create_job(
            "job-hash-1",
            "user-1",
            "text",
            None,
            datetime.now(UTC),
            meeting_date,
            submission,
            raw_key,
            content_hash="hash-abc",
        )
        await repo.update_job_status("job-hash-1", JobStatus.DONE)

        result = await repo.find_job_by_content_hash("hash-abc")
        assert result == "job-hash-1"

    async def test_failed_job_not_returned_by_content_hash(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-hash-2/raw/input.txt"

        await repo.create_job(
            "job-hash-2",
            "user-1",
            "text",
            None,
            datetime.now(UTC),
            meeting_date,
            submission,
            raw_key,
            content_hash="hash-def",
        )
        await repo.fail_job("job-hash-2", "pipeline", "error", recoverable=False)

        result = await repo.find_job_by_content_hash("hash-def")
        assert result is None


class TestRateLimit:
    async def test_count_recent_jobs_per_user(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'

        for i in range(10):
            raw_key = f"jobs/2026-06-01/job-rate-{i}/raw/input.txt"
            await repo.create_job(
                f"job-rate-{i}", "user-rate-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key
            )

        await repo.create_job(
            "job-rate-other",
            "user-rate-2",
            "text",
            None,
            datetime.now(UTC),
            meeting_date,
            submission,
            "jobs/2026-06-01/job-rate-other/raw/input.txt",
        )

        count_user1 = await repo.count_recent_jobs_for_user("user-rate-1")
        count_user2 = await repo.count_recent_jobs_for_user("user-rate-2")

        assert count_user1 == 10
        assert count_user2 == 1


class TestStrandedRecovery:
    async def test_writing_job_returned_as_stranded(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-stranded/raw/input.txt"

        await repo.create_job(
            "job-stranded", "user-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key
        )
        await repo.update_job_status("job-stranded", JobStatus.WRITING)

        stranded = await repo.get_stranded_writing_jobs()
        assert "job-stranded" in stranded

    async def test_done_job_not_returned_as_stranded(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-done/raw/input.txt"

        await repo.create_job("job-done", "user-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key)
        await repo.update_job_status("job-done", JobStatus.DONE)

        stranded = await repo.get_stranded_writing_jobs()
        assert "job-done" not in stranded
