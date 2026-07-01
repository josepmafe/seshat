from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from seshat.models.enums import JobStatus, UserRole

if TYPE_CHECKING:
    import asyncpg


logger = logging.getLogger(__name__)


class ApiKeyNotFoundError(Exception):
    pass


class ApiKeyAlreadyRevokedError(Exception):
    pass


class OpsLedger:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # -- Jobs: Create ----------------------------------------------------------

    async def create_job(
        self,
        job_id: str,
        user_id: str,
        source_type: str,
        idempotency_key: str | None,
        now: datetime,
        meeting_date: date,
        submission_json: str,
        raw_blob_key: str,
        content_hash: str | None = None,
    ) -> None:
        await self._pool.execute(
            "INSERT INTO ops.jobs "
            "(job_id, user_id, status, idempotency_key, source_type, created_at, updated_at, meeting_date, submission, raw_blob_key, content_hash) "  # noqa: E501
            "VALUES ($1, $2, $3, $4, $5, $6, $6, $7, $8, $9, $10)",
            job_id,
            user_id,
            JobStatus.PENDING,
            idempotency_key,
            source_type,
            now,
            meeting_date,
            submission_json,
            raw_blob_key,
            content_hash,
        )

    # -- Jobs: Read -----------------------------------------------------------

    async def get_job(self, job_id: str) -> asyncpg.Record | None:
        return await self._pool.fetchrow("SELECT * FROM ops.jobs WHERE job_id=$1", job_id)

    async def find_job_by_idempotency_key(self, key: str) -> asyncpg.Record | None:
        return await self._pool.fetchrow("SELECT job_id, status FROM ops.jobs WHERE idempotency_key=$1", key)

    async def find_job_by_content_hash(self, content_hash: str) -> str | None:
        return await self._pool.fetchval(
            "SELECT job_id FROM ops.jobs WHERE content_hash=$1 AND status!='failed' ORDER BY created_at DESC LIMIT 1",
            content_hash,
        )

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[asyncpg.Record]:
        if status is not None:
            return await self._pool.fetch(
                "SELECT * FROM ops.jobs WHERE status=$1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                status.value,
                limit,
                offset,
            )
        return await self._pool.fetch(
            "SELECT * FROM ops.jobs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit,
            offset,
        )

    async def count_recent_jobs_for_user(self, user_id: str) -> int:
        return await self._pool.fetchval(
            "SELECT COUNT(*) FROM ops.jobs WHERE user_id=$1 AND created_at > NOW() - INTERVAL '1 hour'",
            user_id,
        )

    async def count_running_jobs(self) -> int:
        return await self._pool.fetchval(
            "SELECT COUNT(*) FROM ops.jobs WHERE status = ANY($1::text[])",
            JobStatus.running_statuses,
        )

    async def get_stranded_writing_jobs(self) -> list[str]:
        rows = await self._pool.fetch("SELECT job_id FROM ops.jobs WHERE status='writing'")
        return [row["job_id"] for row in rows]

    # -- Jobs: Update --------------------------------------------------------

    async def update_job_status(self, job_id: str, status: JobStatus) -> None:
        now = datetime.now(UTC)
        if status.is_terminal:
            await self._pool.execute(
                "UPDATE ops.jobs SET status=$1, updated_at=$2, finished_at=$2 WHERE job_id=$3",
                status.value,
                now,
                job_id,
            )
        else:
            await self._pool.execute(
                "UPDATE ops.jobs SET status=$1, updated_at=$2 WHERE job_id=$3",
                status.value,
                now,
                job_id,
            )

    async def fail_job(
        self,
        job_id: str,
        stage: str,
        reason: str,
        *,
        recoverable: bool,
    ) -> None:
        payload = json.dumps(
            {"stage": stage, "status": "failed", "reason": reason, "recoverable": recoverable, "usage": {}}
        )
        now = datetime.now(UTC)
        await self._pool.execute(
            "UPDATE ops.jobs SET status='failed', error_payload=$1, updated_at=$2, finished_at=$2 WHERE job_id=$3",
            payload,
            now,
            job_id,
        )

    async def reset_failed_job(self, job_id: str) -> None:
        await self._pool.execute(
            "UPDATE ops.jobs SET status='pending', error_payload=NULL, finished_at=NULL, updated_at=$1 WHERE job_id=$2",
            datetime.now(UTC),
            job_id,
        )

    async def set_job_submission(
        self,
        job_id: str,
        meeting_date: date,
        submission_json: str,
        raw_blob_key: str,
    ) -> None:
        await self._pool.execute(
            "UPDATE ops.jobs SET meeting_date=$1, submission=$2, raw_blob_key=$3, updated_at=$4 WHERE job_id=$5",
            meeting_date,
            submission_json,
            raw_blob_key,
            datetime.now(UTC),
            job_id,
        )

    # -- API Keys: Create ------------------------------------------------------

    async def create_api_key(self, key_hash: str, user_id: str, role: UserRole, now: datetime) -> None:
        await self._pool.execute(
            "INSERT INTO ops.api_keys (key_hash, user_id, role, created_at) VALUES ($1, $2, $3, $4)",
            key_hash,
            user_id,
            role.value,
            now,
        )

    # -- API Keys: Read -------------------------------------------------------

    async def get_api_keys(self) -> list[tuple[str, str, str]]:
        rows = await self._pool.fetch("SELECT key_hash, user_id, role FROM ops.api_keys WHERE revoked_at IS NULL")
        return [(row["key_hash"], row["user_id"], row["role"]) for row in rows]

    async def list_api_keys(self) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            "SELECT id, user_id, role, created_at, revoked_at FROM ops.api_keys ORDER BY created_at DESC"
        )

    # -- API Keys: Update ----------------------------------------------------

    async def revoke_api_key(self, key_id: int, now: datetime) -> None:
        row = await self._pool.fetchrow("SELECT revoked_at FROM ops.api_keys WHERE id=$1", key_id)
        if row is None:
            raise ApiKeyNotFoundError(key_id)
        if row["revoked_at"] is not None:
            raise ApiKeyAlreadyRevokedError(key_id)
        await self._pool.execute(
            "UPDATE ops.api_keys SET revoked_at=$1 WHERE id=$2",
            now,
            key_id,
        )

    # -- Lifecycle -----------------------------------------------------------

    async def is_alive(self) -> bool:
        try:
            await self._pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._pool.close()
