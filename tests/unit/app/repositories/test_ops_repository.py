from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.app.repositories.ops_repository import ApiKeyAlreadyRevokedError, ApiKeyNotFoundError, OpsRepository
from seshat.core.models.enums import JobStatus, UserRole


def _make_repo(**store_returns) -> OpsRepository:
    store = MagicMock()
    for method, return_value in store_returns.items():
        setattr(store, method, AsyncMock(return_value=return_value))
    for method in [
        "create_job",
        "get_job",
        "find_job_by_idempotency_key",
        "find_job_by_content_hash",
        "list_jobs",
        "count_recent_jobs_for_user",
        "count_running_jobs",
        "get_stranded_writing_jobs",
        "update_job_status",
        "fail_job",
        "reset_failed_job",
        "set_job_submission",
        "create_api_key",
        "get_api_keys",
        "list_api_keys",
        "revoke_api_key",
        "is_alive",
    ]:
        if not hasattr(store, method) or not isinstance(getattr(store, method), AsyncMock):
            setattr(store, method, AsyncMock(return_value=None))
    return OpsRepository(store)


class TestOpsRepository:
    async def test_create_job_delegates(self):
        repo = _make_repo()
        now = datetime.now(UTC)
        await repo.create_job("job-1", "user-1", "audio", None, now, date(2026, 6, 1), "{}", "raw/key.mp3")
        repo._store.create_job.assert_called_once_with(
            "job-1", "user-1", "audio", None, now, date(2026, 6, 1), "{}", "raw/key.mp3", None
        )

    async def test_get_api_keys_converts_to_tuples(self):
        rows = [{"key_hash": "h1", "user_id": "alice", "role": "reviewer"}]
        repo = _make_repo(get_api_keys=rows)
        assert await repo.get_api_keys() == [("h1", "alice", "reviewer")]

    async def test_list_jobs_with_status_filter(self):
        repo = _make_repo(list_jobs=[{"job_id": "job-1"}])
        rows = await repo.list_jobs(status=JobStatus.DONE)
        assert len(rows) == 1
        repo._store.list_jobs.assert_called_once_with(JobStatus.DONE, None, None, None, 50, 0)

    async def test_create_api_key_delegates(self):
        repo = _make_repo()
        now = datetime.now(UTC)
        await repo.create_api_key("hashed-key", "alice", UserRole.REVIEWER, now)
        repo._store.create_api_key.assert_called_once_with("hashed-key", "alice", UserRole.REVIEWER, now)

    async def test_revoke_api_key_ok(self):
        repo = _make_repo(revoke_api_key="ok")
        await repo.revoke_api_key(1, datetime.now(UTC))  # no exception

    async def test_revoke_api_key_not_found(self):
        repo = _make_repo(revoke_api_key="not_found")
        with pytest.raises(ApiKeyNotFoundError):
            await repo.revoke_api_key(99, datetime.now(UTC))

    async def test_revoke_api_key_already_revoked(self):
        repo = _make_repo(revoke_api_key="already_revoked")
        with pytest.raises(ApiKeyAlreadyRevokedError):
            await repo.revoke_api_key(1, datetime.now(UTC))
