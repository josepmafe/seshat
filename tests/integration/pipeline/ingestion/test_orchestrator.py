from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from seshat.blob_store.s3_store import S3BlobStore
from seshat.config.settings import BlobStoreConfig, TranscriptionConfig
from seshat.models.transcript import TranscriptMetadata
from seshat.pipeline.ingestion.audio_validator import AudioValidationError
from seshat.pipeline.ingestion.orchestrator import IngestionOrchestrator
from tests.integration.conftest import LOCALSTACK_REGION, LOCALSTACK_TEST_BUCKET, SKIP_IF_NO_LOCALSTACK

pytestmark = [pytest.mark.integration, SKIP_IF_NO_LOCALSTACK]


@pytest.fixture
async def blob_store(localstack_s3_url):
    config = BlobStoreConfig(
        bucket=LOCALSTACK_TEST_BUCKET,
        region=LOCALSTACK_REGION,
        endpoint_url=localstack_s3_url,
    )
    store = S3BlobStore(config)
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def mock_transcriber():
    svc = MagicMock()
    svc.transcribe = AsyncMock(return_value="We decided to use PostgreSQL.")
    return svc


class TestIngestionOrchestratorAudio:
    async def test_ingest_valid_mp3(self, blob_store, mock_transcriber):
        service = IngestionOrchestrator(mock_transcriber, blob_store, TranscriptionConfig())
        audio_bytes = b"ID3" + b"\x00" * 30
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        doc = await service.ingest_audio(audio_bytes, date(2026, 4, 21), "job-audio-1", metadata)
        assert doc.source_type == "audio"
        assert doc.blob_key == "jobs/2026-04-21/job-audio-1/raw/transcript.txt"

    async def test_ingest_with_matching_filename(self, blob_store, mock_transcriber):
        service = IngestionOrchestrator(mock_transcriber, blob_store, TranscriptionConfig())
        audio_bytes = b"ID3" + b"\x00" * 30
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        doc = await service.ingest_audio(
            audio_bytes, date(2026, 4, 21), "job-audio-2", metadata, filename="recording.mp3"
        )
        assert doc.blob_key == "jobs/2026-04-21/job-audio-2/raw/transcript.txt"

    async def test_ingest_mismatched_filename_raises(self, blob_store, mock_transcriber):
        service = IngestionOrchestrator(mock_transcriber, blob_store, TranscriptionConfig())
        audio_bytes = b"ID3" + b"\x00" * 30
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        with pytest.raises(AudioValidationError, match="mismatch"):
            await service.ingest_audio(
                audio_bytes, date(2026, 4, 21), "job-mismatch", metadata, filename="recording.wav"
            )

    async def test_ingest_oversized_raises(self, blob_store, mock_transcriber):
        service = IngestionOrchestrator(mock_transcriber, blob_store, TranscriptionConfig(max_file_bytes=10))
        with pytest.raises(AudioValidationError, match="exceeds maximum"):
            await service.ingest_audio(b"ID3" + b"\x00" * 30, date(2026, 4, 21), "job-oversized", MagicMock())


class TestIngestionOrchestratorText:
    async def test_ingest_valid_yaml(self, blob_store, mock_transcriber):
        service = IngestionOrchestrator(mock_transcriber, blob_store, TranscriptionConfig())
        raw = yaml.dump(
            {
                "date": "2026-04-21",
                "content": "We decided to use PostgreSQL.",
                "participants": ["Alice"],
            }
        ).encode()
        doc = await service.ingest_text(raw, "meeting.yaml", "job-text-1")
        assert doc.source_type == "text"
        assert doc.blob_key == "jobs/2026-04-21/job-text-1/raw/transcript.txt"
        assert doc.metadata.participants == ["Alice"]
