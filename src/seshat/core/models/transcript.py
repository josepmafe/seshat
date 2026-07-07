from datetime import date, timedelta
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from seshat.core.models.base import SeshatModel


class TranscriptMetadata(SeshatModel):
    meeting_date: date
    participants: list[str] | None = None
    duration: timedelta | None = Field(default=None, gt=0)
    language: str = Field(default="en", min_length=2, description="BCP-47 language code for the transcript.")


class TranscriptDocument(SeshatModel):
    id: UUID = Field(default_factory=uuid4)
    idempotency_key: str | None = Field(
        default=None, description="Client-supplied key to detect duplicate submissions."
    )
    schema_version: str = Field(
        default="1.0", pattern=r"^\d+\.\d+$", description="Schema version for forward-compatibility checks."
    )
    source_type: Literal["audio", "text"] = Field(
        description="Whether the source was audio (transcribed) or pre-existing text."
    )
    blob_key: str = Field(description="Blob storage key where the transcript text is stored.")
    metadata: TranscriptMetadata
