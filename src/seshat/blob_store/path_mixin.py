from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date


class BlobPathsMixin:
    def raw_input_key(self, meeting_date: date, job_id: str, extension: str) -> str:
        return f"jobs/{meeting_date.isoformat()}/{job_id}/raw/input.{extension}"

    def raw_transcript_key(self, meeting_date: date, job_id: str) -> str:
        return f"jobs/{meeting_date.isoformat()}/{job_id}/raw/transcript.txt"

    def curated_extraction_key(self, meeting_date: date, job_id: str) -> str:
        return f"jobs/{meeting_date.isoformat()}/{job_id}/curated/extraction.json"

    def init_source_key(self, init_job_id: str, index: int) -> str:
        return f"init/{init_job_id}/source/{index:04d}.md"

    def init_curated_extraction_key(self, init_job_id: str) -> str:
        return f"init/{init_job_id}/curated/extraction.json"
