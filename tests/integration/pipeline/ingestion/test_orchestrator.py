from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from seshat.config.settings import TranscriptionConfig
from seshat.models.transcript import TranscriptMetadata
from seshat.pipeline.ingestion.audio_validator import AudioValidationError
from seshat.pipeline.ingestion.orchestrator import IngestionOrchestrator
from tests.integration.conftest import SKIP_IF_NO_LOCALSTACK

pytestmark = [pytest.mark.integration, SKIP_IF_NO_LOCALSTACK]


@pytest.fixture
def mock_transcriber():
    transcriber = MagicMock()
    transcriber.transcribe = AsyncMock(return_value="We decided to use PostgreSQL.")
    return transcriber


def _build_orchestrator(mock_transcriber, blob_store, transcription_config=None):
    return IngestionOrchestrator(mock_transcriber, blob_store, transcription_config or TranscriptionConfig())


@pytest.fixture
def orchestrator(mock_transcriber, blob_store):
    return _build_orchestrator(mock_transcriber, blob_store)


class TestIngestionOrchestratorAudio:
    async def test_ingest_valid_mp3(self, orchestrator, short_audio_bytes):
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        doc = await orchestrator.ingest_audio(short_audio_bytes, date(2026, 4, 21), "job-audio-1", metadata)
        assert doc.source_type == "audio"
        assert doc.blob_key == "jobs/2026-04-21/job-audio-1/raw/transcript.txt"

    async def test_ingest_with_matching_filename(self, orchestrator, short_audio_bytes):
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        doc = await orchestrator.ingest_audio(
            short_audio_bytes, date(2026, 4, 21), "job-audio-2", metadata, filename="recording.mp3"
        )
        assert doc.blob_key == "jobs/2026-04-21/job-audio-2/raw/transcript.txt"

    async def test_ingest_mismatched_filename_raises(self, orchestrator, short_audio_bytes):
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        with pytest.raises(AudioValidationError, match="mismatch"):
            await orchestrator.ingest_audio(
                short_audio_bytes, date(2026, 4, 21), "job-mismatch", metadata, filename="recording.wav"
            )

    async def test_ingest_oversized_raises(self, mock_transcriber, blob_store, short_audio_bytes):
        orchestrator = _build_orchestrator(mock_transcriber, blob_store, TranscriptionConfig(max_file_bytes=10))
        with pytest.raises(AudioValidationError, match="exceeds maximum"):
            await orchestrator.ingest_audio(short_audio_bytes, date(2026, 4, 21), "job-oversized", MagicMock())


class TestIngestionOrchestratorText:
    async def test_ingest_valid_yaml(self, orchestrator):
        raw = yaml.dump(
            {
                "date": "2026-04-21",
                "content": "We decided to use PostgreSQL.",
                "participants": ["Alice"],
            }
        ).encode()
        doc = await orchestrator.ingest_text(raw, "meeting.yaml", "job-text-1")
        assert doc.source_type == "text"
        assert doc.blob_key == "jobs/2026-04-21/job-text-1/raw/transcript.txt"
        assert doc.metadata.participants == ["Alice"]
