from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from seshat.api.dependencies import CurrentUser, get_app_state, require_role
from seshat.api.state import AppState
from seshat.models.api_jobs import ApproveRequest, BulkApproveRule, NodeDecision, RateLimitError
from seshat.models.api_responses import JobActionResponse, JobSubmitResponse
from seshat.models.enums import ApprovalMethod, JobStatus, NodeStatus, UserRole
from seshat.models.jobs import JobResponse
from seshat.models.nodes import ExtractionResult, KBNode
from seshat.models.submission import JobSubmissionRequest

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_role(UserRole.VIEWER))])


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=None)
async def submit_job(
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.REVIEWER))],
    file: Annotated[UploadFile, ...],
    body: Annotated[str, Form()],
) -> JobSubmitResponse | JSONResponse:
    submission = JobSubmissionRequest.model_validate_json(body)

    if submission.idempotency_key:
        existing = await state.ops.find_job_by_idempotency_key(submission.idempotency_key)
        if existing and existing["status"] != "failed":
            return JobSubmitResponse(job_id=existing["job_id"])

    if await state.ops.count_recent_jobs_for_user(user.user_id) >= state.config.max_jobs_per_user_per_hour:
        return JSONResponse(
            status_code=429,
            content=RateLimitError(limit_type="per_user_hourly_cap").model_dump(),
        )

    if await state.ops.count_running_jobs() >= state.config.max_concurrent_jobs:
        return JSONResponse(
            status_code=429,
            content=RateLimitError(limit_type="global_concurrency_cap").model_dump(),
        )

    if submission.overrides is not None and not user.role.is_at_least(UserRole.OPERATOR):
        raise HTTPException(status_code=403, detail="Config overrides require operator role")

    job_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    await state.ops.create_job(job_id, user.user_id, submission.source_type, submission.idempotency_key, now)

    file_bytes = await file.read()
    await state.queue.enqueue(job_id, state.runner.run, job_id, file_bytes, submission)

    return JobSubmitResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> JobResponse:
    row = await state.ops.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    elapsed = (datetime.now(UTC) - row["created_at"]).total_seconds()
    error = json.loads(row["error_payload"]) if row["error_payload"] else None
    return JobResponse(
        job_id=job_id,
        status=JobStatus(row["status"]),
        idempotency_key=row["idempotency_key"],
        current_stage=JobStatus(row["status"]) if row["status"] not in ("done", "failed") else None,
        stage_progress=None,
        elapsed_seconds=elapsed,
        error=error,
        mlflow_run_id=row["mlflow_run_id"],
    )


@router.get("/{job_id}/results")
async def get_job_results(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> ExtractionResult:
    row = await state.ops.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] not in ("awaiting_review", "done"):
        raise HTTPException(status_code=409, detail="Results not yet available")

    result = state.results.get(job_id)
    if not result:
        # TODO: fall back to reading curated/extraction.json from blob storage when in-memory
        # result is absent (e.g. after server restart). Requires: (1) pipeline_runner writes
        # the blob at the start of WRITING, (2) meeting_date is stored in ops.jobs so the
        # blob key can be reconstructed from the job row alone. Deferred to a follow-up PR.
        raise HTTPException(status_code=404, detail="Extraction result not found (server may have restarted)")

    return result


@router.post("/{job_id}/approve")
async def approve_job(
    job_id: str,
    approve_request: ApproveRequest,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.REVIEWER))],
) -> JobActionResponse:
    row = await state.ops.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "awaiting_review":
        raise HTTPException(status_code=409, detail="Job is not awaiting review")

    result = state.results.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Extraction result not found")

    now = datetime.now(UTC)
    nodes = list(result.nodes)

    if approve_request.approve_above_threshold:
        nodes = _apply_bulk_rule(nodes, approve_request.approve_above_threshold, user.user_id, now)

    if approve_request.decisions:
        nodes = _apply_decisions(nodes, approve_request.decisions, user.user_id, now)

    state.results[job_id] = result.model_copy(update={"nodes": nodes})

    await state.ops.update_job_status(job_id, JobStatus.WRITING)
    await state.queue.enqueue(job_id, state.runner.run_post_approval, job_id)

    return JobActionResponse(status="accepted")


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> JobActionResponse:
    row = await state.ops.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "failed":
        raise HTTPException(status_code=409, detail="Only failed jobs can be retried")

    # TODO: File bytes are not yet persisted to blob storage on submit, so re-running
    # the full pipeline from just the job ID is not possible. Use POST /jobs with
    # the original idempotency_key for now (creates a new job ID but achieves the
    # same result). Tracked for implementation in a follow-up PR.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Retry-by-job-id not yet implemented. Re-submit via POST /jobs with the original idempotency_key.",
    )


def _apply_bulk_rule(nodes: list[KBNode], rule: BulkApproveRule, user_id: str, now: datetime) -> list[KBNode]:
    exclude = set(rule.exclude or [])
    result = []
    for node in nodes:
        node_id = str(node.id)
        if node.status == NodeStatus.PENDING_REVIEW and node.confidence >= rule.threshold and node_id not in exclude:
            metadata = node.metadata._with(approval_method=ApprovalMethod.BULK, approved_by=user_id, approved_at=now)
            node = node._with(status=NodeStatus.APPROVED, metadata=metadata)
        result.append(node)
    return result


def _apply_decisions(nodes: list[KBNode], decisions: list[NodeDecision], user_id: str, now: datetime) -> list[KBNode]:
    node_map = {str(n.id): n for n in nodes}
    for decision in decisions:
        node = node_map.get(decision.node_id)
        if not node:
            continue

        if decision.action == "approve":
            node_kwargs: dict[str, Any] = {}
            meta_kwargs: dict[str, Any] = {
                "approval_method": ApprovalMethod.INDIVIDUAL,
                "approved_by": user_id,
                "approved_at": now,
            }

            if edited_content := decision.edited_content:
                node_kwargs |= {"title": edited_content.title, "description": edited_content.description}
                meta_kwargs |= {"corrected_by": user_id, "corrected_at": now}

            node_kwargs |= {"status": NodeStatus.APPROVED, "metadata": node.metadata._with(**meta_kwargs)}
            node_map[decision.node_id] = node._with(**node_kwargs)

        elif decision.action == "reject":
            node_map[decision.node_id] = node._with(status=NodeStatus.REJECTED)

        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {decision.action}")

    return list(node_map.values())
