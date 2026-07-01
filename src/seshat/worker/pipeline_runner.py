from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from seshat.models.enums import ApprovalMethod, JobStatus, NodeStatus
from seshat.models.nodes import ExtractionResult
from seshat.utils.log import get_logger, set_job_id

if TYPE_CHECKING:
    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.models.nodes import IdentificationResult, KBRelationship
    from seshat.models.submission import JobSubmissionRequest
    from seshat.ops.ledger import OpsLedger
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.pipeline.ingestion.orchestrator import IngestionOrchestrator
    from seshat.worker.bootstrap import WorkerContext
    from seshat.worker.writing_stage import WritingStage

logger = get_logger(__name__)


class PipelineRunner:
    def __init__(
        self,
        ingestion_orchestrator: IngestionOrchestrator,
        extraction_orchestrator: ExtractionOrchestrator,
        writing_stage: WritingStage,
        ops_ledger: OpsLedger,
        blob_store: S3BlobStore,
        *,
        extraction_auto_mode: bool = False,
    ) -> None:
        self._ingestion = ingestion_orchestrator
        self._extraction = extraction_orchestrator
        self._writing = writing_stage
        self._ops = ops_ledger
        self._blob_store = blob_store
        self._extraction_auto_mode = extraction_auto_mode
        self._pending: dict[str, IdentificationResult] = {}
        self._results: dict[str, ExtractionResult] = {}

    @classmethod
    def from_context(cls, ctx: WorkerContext) -> PipelineRunner:
        return cls(
            ingestion_orchestrator=ctx.ingestion_orchestrator,
            extraction_orchestrator=ctx.extraction_orchestrator,
            writing_stage=ctx.writing_stage,
            ops_ledger=ctx.ops,
            blob_store=ctx.blob_store,
            extraction_auto_mode=ctx.extraction_orchestrator._config.auto_mode,
        )

    @property
    def results(self) -> dict[str, ExtractionResult]:
        return self._results

    async def run(
        self, job_id: str, file_bytes: bytes, submission: JobSubmissionRequest, user_id: str | None = None
    ) -> None:
        """Convenience method: run pre-approval and, if no review needed, post-approval."""
        identification_result = await self._run_pre_approval(job_id, file_bytes, submission, user_id=user_id)
        if identification_result is None:
            return

        # In auto_mode (service config or per-request override) all nodes are APPROVED or REJECTED
        # by _apply_auto_mode, so has_pending is always False and post-approval fires immediately.
        has_pending = any(n.status == NodeStatus.PENDING_REVIEW for n in identification_result.nodes)
        if not has_pending:
            await self.run_post_approval(job_id)

    async def _run_pre_approval(
        self,
        job_id: str,
        file_bytes: bytes,
        submission: JobSubmissionRequest,
        user_id: str | None = None,
    ) -> IdentificationResult | None:
        """Ingest and identify. Stores result; sets AWAITING_REVIEW. Returns None on failure."""
        try:
            set_job_id(job_id)
            await self._ops.update_job_status(job_id, JobStatus.TRANSCRIBING)

            if submission.source_type == "audio":
                doc = await self._ingestion.ingest_audio(
                    file_bytes,
                    submission.metadata.meeting_date,
                    job_id,
                    submission.metadata,
                )
            else:
                doc = await self._ingestion.ingest_text(
                    file_bytes,
                    submission.metadata.meeting_date,
                    job_id,
                    "input.yaml",
                )

            await self._ops.update_job_status(job_id, JobStatus.EXTRACTING)
            identification_result = await self._extraction.run_identification(doc, job_id, user_id=user_id)

            if self._effective_auto_mode(submission):
                identification_result = self._apply_auto_mode(identification_result, datetime.now(UTC))

            self._pending[job_id] = identification_result
            # Pre-populate results so GET /jobs/{id}/results works during AWAITING_REVIEW.
            # run_post_approval overwrites this with the complete result (including relationships).
            await self._store_result(job_id, identification_result, [])

            await self._ops.update_job_status(job_id, JobStatus.AWAITING_REVIEW)
            return identification_result

        except Exception as exc:
            logger.exception("Job failed (pre-approval): %s", exc)
            # MVP: all failures marked recoverable; operator reads error_payload to triage.
            # Revisit when a retry queue is added — some failures (e.g. bad input) are permanent.
            await self._ops.fail_job(job_id, "pre_approval", str(exc), recoverable=True)
            return None

    def _effective_auto_mode(self, submission: JobSubmissionRequest) -> bool:
        if submission.auto_mode:
            return True

        overrides = submission.overrides
        if overrides and overrides.extraction and overrides.extraction.auto_mode:
            return True

        return self._extraction_auto_mode

    def _apply_auto_mode(self, identification_result: IdentificationResult, now: datetime) -> IdentificationResult:
        """Promote PENDING_REVIEW nodes to APPROVED when auto_mode is active."""
        updated_nodes = []
        for node in identification_result.nodes:
            if node.status == NodeStatus.PENDING_REVIEW:
                metadata = node.metadata._with(
                    approval_method=ApprovalMethod.AUTO,
                    approved_at=now,
                )
                node = node._with(status=NodeStatus.APPROVED, metadata=metadata)
            updated_nodes.append(node)

        return identification_result._with(nodes=updated_nodes)

    async def run_post_approval(self, job_id: str) -> None:
        """Resolve, write, and mark DONE. Expects _run_pre_approval to have completed."""
        try:
            identification_result = self._pending.pop(job_id)
            set_job_id(job_id)

            approved = [n for n in identification_result.nodes if n.status == NodeStatus.APPROVED]
            resol = await self._extraction.run_resolution(job_id, approved=approved)

            result = await self._store_result(job_id, identification_result, resol.relationships)

            await self._ops.update_job_status(job_id, JobStatus.WRITING)
            written = await self._writing.write(result)

            await self._ops.update_job_status(job_id, JobStatus.DONE)
            logger.info("Job done: %d nodes written", written)

        except Exception as exc:
            logger.exception("Job failed (post-approval): %s", exc)
            # MVP: all failures marked recoverable; see pre_approval comment above.
            await self._ops.fail_job(job_id, "post_approval", str(exc), recoverable=True)

    async def _store_result(
        self,
        job_id: str,
        identification_result: IdentificationResult,
        relationships: list[KBRelationship],
    ) -> ExtractionResult:
        result = ExtractionResult(
            job_id=job_id,
            nodes=identification_result.nodes,
            relationships=relationships,
            confidence_breakdowns={str(k): v for k, v in identification_result.confidence_breakdowns.items()},
        )
        self._results[job_id] = result

        job_row = await self._ops.get_job(job_id)
        meeting_date = job_row["meeting_date"] if job_row else None
        if meeting_date is not None:
            await self._blob_store.put(
                self._blob_store.curated_extraction_key(meeting_date, job_id),
                result.model_dump_json().encode(),
            )
        else:
            logger.warning("Job has no meeting_date; skipping `extraction.json` blob write")

        return result
