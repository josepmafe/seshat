from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.core.models.transcript import TranscriptDocument, TranscriptMetadata
from seshat.core.utils.log import get_logger
from seshat.observability.usage_tracker import track_token_budget
from seshat.pipeline.ingestion.audio_validator import AudioValidator
from seshat.pipeline.ingestion.text_validator import TextValidationError, TextValidator

if TYPE_CHECKING:
    from datetime import date

    from seshat.core.config.settings import TranscriptionConfig
    from seshat.repositories.blob_repository import BlobRepository
    from seshat.transcription.base import AbstractTranscriber

logger = get_logger(__name__)


class IngestionOrchestrator:
    def __init__(
        self,
        transcriber: AbstractTranscriber,
        blob_repo: BlobRepository,
        transcription_config: TranscriptionConfig,
    ) -> None:
        self._transcription = transcriber
        self._blob = blob_repo
        self._config = transcription_config

    @track_token_budget("ingestion", uncapped=True)
    async def ingest_audio(
        self,
        audio_bytes: bytes,
        meeting_date: date,
        job_id: str,
        metadata: TranscriptMetadata,
        filename: str | None = None,
    ) -> TranscriptDocument:
        ext = self._validate_and_get_audio_extension(audio_bytes, filename)

        transcript_text = await self._transcription.transcribe(audio_bytes, extension=ext)

        await self._blob.put_raw_transcript(meeting_date, job_id, transcript_text.encode())
        transcript_key = self._blob.raw_transcript_key(meeting_date, job_id)
        logger.info("Job: uploaded transcript to %s", transcript_key)

        return TranscriptDocument(
            source_type="audio",
            blob_key=transcript_key,
            metadata=metadata,
        )

    async def ingest_text(
        self,
        raw_bytes: bytes,
        meeting_date: date,
        job_id: str,
        filename: str,
    ) -> TranscriptDocument:
        parsed = TextValidator.parse(raw_bytes, filename)

        if parsed.meeting_date != meeting_date:
            raise TextValidationError(
                f"meeting_date mismatch: submission says {meeting_date}, file says {parsed.meeting_date}"
            )

        await self._blob.put_raw_transcript(meeting_date, job_id, parsed.content.encode())
        transcript_key = self._blob.raw_transcript_key(meeting_date, job_id)
        logger.info("Job: uploaded transcript to %s", transcript_key)

        metadata = TranscriptMetadata(
            meeting_date=parsed.meeting_date,
            participants=parsed.participants,
        )
        return TranscriptDocument(
            source_type="text",
            blob_key=transcript_key,
            metadata=metadata,
        )

    def _validate_and_get_audio_extension(self, audio_bytes: bytes, filename: str | None) -> str:
        AudioValidator.check_size(len(audio_bytes), self._config.max_file_bytes)

        audio_duration = AudioValidator.get_duration_seconds(audio_bytes)
        AudioValidator.check_duration(audio_duration, self._config.max_audio_seconds)

        alleged_ext = filename.rsplit(".", 1)[-1].lower() if filename and "." in filename else None
        ext = AudioValidator.validate_magic_bytes(audio_bytes, alleged_ext=alleged_ext)
        return ext
