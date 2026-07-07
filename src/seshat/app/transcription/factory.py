from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.core.models.enums import TranscriptionProvider
from seshat.core.utils.log import get_logger
from seshat.infra.secrets.factory import get_secrets_resolver
from seshat.observability.usage_tracker import TrackingTranscriber

if TYPE_CHECKING:
    from seshat.app.transcription.base import AbstractTranscriber
    from seshat.core.config.settings import SeshatConfig

logger = get_logger(__name__)


def get_transcriber(
    config: SeshatConfig,
) -> AbstractTranscriber:
    secrets = get_secrets_resolver(config)
    api_key = secrets.get_secret(config.transcription.api_key_secret_key)  # type: ignore[arg-type]

    logger.debug("Initialising transcriber: %s", config.transcription.provider)

    raw: AbstractTranscriber
    match config.transcription.provider:
        case TranscriptionProvider.ASSEMBLYAI:
            from seshat.app.transcription.assemblyai_transcriber import AssemblyAITranscriber

            raw = AssemblyAITranscriber(config.transcription, api_key)
        case TranscriptionProvider.OPENAI:
            from seshat.app.transcription.openai_transcriber import OpenAITranscriber

            raw = OpenAITranscriber(config.transcription, api_key)
        case _:
            raise ValueError(f"Unsupported transcription provider: {config.transcription.provider}")

    return TrackingTranscriber(raw)
