from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from seshat.config.settings import TranscriptionConfig
from seshat.models.enums import TranscriptionProvider
from seshat.transcription.assemblyai_transcriber import AssemblyAITranscriber
from seshat.transcription.factory import get_transcriber
from seshat.transcription.openai_transcriber import OpenAITranscriber

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig


class TestGetTranscriber:
    def test_returns_assemblyai_transcriber(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        service = get_transcriber(minimal_config)
        assert isinstance(service, AssemblyAITranscriber)

    def test_returns_openai_transcriber(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        config = minimal_config.model_copy(
            update={"transcription": TranscriptionConfig(provider=TranscriptionProvider.OPENAI)}
        )
        service = get_transcriber(config)
        assert isinstance(service, OpenAITranscriber)

    def test_unsupported_provider_raises(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        config = minimal_config.model_copy(
            update={"transcription": TranscriptionConfig(provider=TranscriptionProvider.DEEPGRAM)}
        )
        with pytest.raises(ValueError, match="Unsupported transcription provider"):
            get_transcriber(config)
