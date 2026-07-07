from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from seshat.app.transcription.assemblyai_transcriber import AssemblyAITranscriber
from seshat.app.transcription.factory import get_transcriber
from seshat.app.transcription.openai_transcriber import OpenAITranscriber
from seshat.core.config.settings import TranscriptionConfig
from seshat.core.models.enums import TranscriptionProvider
from seshat.observability.usage_tracker import TrackingTranscriber

if TYPE_CHECKING:
    from seshat.core.config.settings import SeshatConfig


class TestGetTranscriber:
    def test_returns_assemblyai_transcriber(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        transcriber = get_transcriber(minimal_config)
        assert isinstance(transcriber, TrackingTranscriber)
        assert isinstance(transcriber._transcriber, AssemblyAITranscriber)

    def test_returns_openai_transcriber(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        config = minimal_config.model_copy(
            update={"transcription": TranscriptionConfig(provider=TranscriptionProvider.OPENAI)}
        )
        transcriber = get_transcriber(config)
        assert isinstance(transcriber, TrackingTranscriber)
        assert isinstance(transcriber._transcriber, OpenAITranscriber)

    def test_unsupported_provider_raises(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        config = minimal_config.model_copy(
            update={"transcription": TranscriptionConfig(provider=TranscriptionProvider.DEEPGRAM)}
        )
        with pytest.raises(ValueError, match="Unsupported transcription provider"):
            get_transcriber(config)
