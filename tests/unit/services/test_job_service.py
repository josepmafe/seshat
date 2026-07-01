from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.models.enums import JobStatus, NodeStatus
from seshat.models.nodes import ExtractionResult, IdentificationResult, ResolutionResult
from seshat.services.job_service import (
    JobNotFoundError,
    JobService,
    JobStateError,
)
from tests.helpers import make_node


def _make_service(
    nodes: list | None = None,
) -> tuple[JobService, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    nodes = nodes or [make_node()]

    ident_result = IdentificationResult(job_id="job-1", nodes=nodes, confidence_breakdowns={})
    resol_result = ResolutionResult(job_id="job-1", relationships=[])

    ingestion = MagicMock()
    ingestion.ingest_text = AsyncMock(return_value=MagicMock())
    ingestion.ingest_audio = AsyncMock(return_value=MagicMock())

    extraction = MagicMock()
    extraction.run_identification = AsyncMock(return_value=ident_result)
    extraction.run_resolution = AsyncMock(return_value=resol_result)
    extraction._config = MagicMock()
    extraction._config.auto_mode = False

    node_repo = MagicMock()
    node_repo.write_batch = AsyncMock(return_value=len([n for n in nodes if n.status == NodeStatus.APPROVED]))
    node_repo.paginated_query = AsyncMock(return_value=[])
    node_repo.delete_node = AsyncMock()

    ops = MagicMock()
    ops.update_job_status = AsyncMock()
    ops.fail_job = AsyncMock()
    ops.get_job = AsyncMock(return_value=None)
    ops.count_recent_jobs_for_user = AsyncMock(return_value=0)
    ops.count_running_jobs = AsyncMock(return_value=0)
    ops.find_job_by_idempotency_key = AsyncMock(return_value=None)
    ops.find_job_by_content_hash = AsyncMock(return_value=None)
    ops.create_job = AsyncMock()
    ops.reset_failed_job = AsyncMock()
    ops.get_stranded_writing_jobs = AsyncMock(return_value=[])

    blob = MagicMock()
    blob.put_by_key = AsyncMock()
    blob.get_by_key = AsyncMock(return_value=None)
    blob.put_curated_extraction = AsyncMock()
    blob.get_curated_extraction = AsyncMock(return_value=None)
    blob.raw_input_key = MagicMock(return_value="raw/key")

    config = MagicMock()
    config.api.max_jobs_per_user_per_hour = 10
    config.api.max_concurrent_jobs = 5

    queue = MagicMock()
    queue.enqueue = AsyncMock()

    svc = JobService(config, ops, blob, node_repo, extraction, ingestion, queue)
    return svc, ingestion, extraction, node_repo, ops, blob


def _make_submission(source_type: str = "text") -> MagicMock:
    sub = MagicMock()
    sub.source_type = source_type
    sub.auto_mode = False
    sub.overrides = None
    sub.idempotency_key = None
    sub.force = False
    sub.metadata.meeting_date = date(2026, 1, 15)
    sub.model_dump_json = MagicMock(return_value="{}")
    return sub


class TestPreApproval:
    async def test_text_job_sets_statuses_and_parks_result(self):
        svc, _, _, _, ops, _ = _make_service()
        ident = await svc._run_pre_approval("job-1", b"data", _make_submission())

        assert ident is not None
        assert "job-1" in svc._pending
        ops.update_job_status.assert_any_await("job-1", JobStatus.TRANSCRIBING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.EXTRACTING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.AWAITING_REVIEW)

    async def test_audio_job_calls_ingest_audio(self):
        svc, ingestion, _, _, _, _ = _make_service()
        await svc._run_pre_approval("job-1", b"audio", _make_submission("audio"))
        ingestion.ingest_audio.assert_called_once()
        ingestion.ingest_text.assert_not_called()

    async def test_failure_calls_fail_job_and_returns_none(self):
        svc, ingestion, _, _, ops, _ = _make_service()
        ingestion.ingest_text.side_effect = RuntimeError("network error")

        result = await svc._run_pre_approval("job-1", b"data", _make_submission())

        assert result is None
        ops.fail_job.assert_called_once()
        assert "job-1" not in svc._pending


class TestPostApproval:
    async def test_resolves_writes_and_sets_done(self):
        svc, _, extraction, node_repo, ops, _ = _make_service()
        svc._pending["job-1"] = await extraction.run_identification(MagicMock(), "job-1")

        await svc._run_post_approval("job-1")

        extraction.run_resolution.assert_called_once()
        node_repo.write_batch.assert_called_once()
        ops.update_job_status.assert_any_await("job-1", JobStatus.WRITING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.DONE)
        assert "job-1" in svc._results
        assert isinstance(svc._results["job-1"], ExtractionResult)

    async def test_passes_approved_nodes_to_resolution(self):
        approved = make_node()
        rejected = make_node("rejected", status=NodeStatus.REJECTED)
        svc, _, extraction, _, _, _ = _make_service(nodes=[approved, rejected])
        svc._pending["job-1"] = IdentificationResult(
            job_id="job-1", nodes=[approved, rejected], confidence_breakdowns={}
        )

        await svc._run_post_approval("job-1")

        _, kwargs = extraction.run_resolution.call_args
        assert kwargs["approved"] == [approved]

    async def test_writes_curated_extraction_blob(self):
        meeting_date = date(2026, 1, 1)
        svc, _, extraction, _, ops, blob = _make_service()
        ops.get_job = AsyncMock(return_value={"meeting_date": meeting_date})
        svc._pending["job-1"] = await extraction.run_identification(MagicMock(), "job-1")

        await svc._run_post_approval("job-1")

        blob.put_curated_extraction.assert_called_once()
        call_args = blob.put_curated_extraction.call_args
        assert call_args.args[0] == meeting_date
        assert call_args.args[1] == "job-1"

    async def test_failure_calls_fail_job(self):
        svc, _, extraction, _, ops, _ = _make_service()
        svc._pending["job-1"] = IdentificationResult(job_id="job-1", nodes=[], confidence_breakdowns={})
        extraction.run_resolution.side_effect = RuntimeError("llm timeout")

        await svc._run_post_approval("job-1")

        ops.fail_job.assert_called_once()
        call_kwargs = ops.fail_job.call_args
        assert "post_approval" in call_kwargs[0]


class TestAutoMode:
    async def test_auto_mode_field_promotes_pending_and_fires_post_approval(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        svc, _, extraction, node_repo, _, _ = _make_service(nodes=[node])

        sub = _make_submission()
        sub.auto_mode = True

        await svc._run("job-1", b"data", sub)

        extraction.run_resolution.assert_called_once()
        node_repo.write_batch.assert_called_once()
        approved = [n for n in svc._results["job-1"].nodes if n.status == NodeStatus.APPROVED]
        assert len(approved) == 1

    async def test_auto_mode_via_extraction_override_also_works(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        svc, _, extraction, node_repo, _, _ = _make_service(nodes=[node])

        sub = _make_submission()
        sub.overrides = MagicMock()
        sub.overrides.extraction = MagicMock()
        sub.overrides.extraction.auto_mode = True

        await svc._run("job-1", b"data", sub)

        extraction.run_resolution.assert_called_once()
        node_repo.write_batch.assert_called_once()

    async def test_pending_nodes_stay_without_auto_mode(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        svc, _, extraction, node_repo, _, _ = _make_service(nodes=[node])

        await svc._run("job-1", b"data", _make_submission())

        extraction.run_resolution.assert_not_called()
        node_repo.write_batch.assert_not_called()


class TestApprove:
    async def test_not_found_raises(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value=None)

        with pytest.raises(JobNotFoundError):
            await svc.approve("job-1", MagicMock(), "alice")

    async def test_wrong_state_raises(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value={"status": "pending"})

        with pytest.raises(JobStateError):
            await svc.approve("job-1", MagicMock(), "alice")

    async def test_missing_result_raises(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value={"status": JobStatus.AWAITING_REVIEW})

        with pytest.raises(JobNotFoundError):
            await svc.approve("job-1", MagicMock(), "alice")


class TestRecoverStranded:
    async def test_marks_stranded_jobs_failed(self):
        svc, *_ = _make_service()
        svc._ops.get_stranded_writing_jobs = AsyncMock(return_value=["job-a", "job-b"])

        await svc.recover_stranded()

        assert svc._ops.fail_job.call_count == 2
        svc._ops.fail_job.assert_any_call("job-a", JobStatus.WRITING, "Server crash during write", recoverable=True)
        svc._ops.fail_job.assert_any_call("job-b", JobStatus.WRITING, "Server crash during write", recoverable=True)
