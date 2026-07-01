from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from seshat.blob_store.s3_store import S3BlobStore


class BlobRepository:
    """S3-backed blob repository. Owns key conventions — callers never construct key strings."""

    def __init__(self, store: S3BlobStore) -> None:
        self._store = store

    async def connect(self) -> None:
        await self._store.connect()

    async def close(self) -> None:
        await self._store.close()

    async def put_by_key(self, key: str, data: bytes) -> None:
        await self._store.put(key, data)

    async def get_by_key(self, key: str) -> bytes | None:
        return await self._store.get(key)

    async def put_raw_input(self, meeting_date: date, job_id: str, extension: str, data: bytes) -> None:
        key = self.raw_input_key(meeting_date, job_id, extension)
        await self.put_by_key(key, data)

    async def get_raw_input(self, meeting_date: date, job_id: str, extension: str) -> bytes | None:
        key = self.raw_input_key(meeting_date, job_id, extension)
        return await self.get_by_key(key)

    async def put_raw_transcript(self, meeting_date: date, job_id: str, data: bytes) -> None:
        key = self.raw_transcript_key(meeting_date, job_id)
        await self.put_by_key(key, data)

    async def get_raw_transcript(self, meeting_date: date, job_id: str) -> bytes | None:
        key = self.raw_transcript_key(meeting_date, job_id)
        return await self.get_by_key(key)

    async def put_curated_extraction(self, meeting_date: date, job_id: str, data: bytes) -> None:
        key = self._curated_extraction_key(meeting_date, job_id)
        await self.put_by_key(key, data)

    async def get_curated_extraction(self, meeting_date: date, job_id: str) -> bytes | None:
        key = self._curated_extraction_key(meeting_date, job_id)
        return await self.get_by_key(key)

    @staticmethod
    def raw_input_key(meeting_date: date, job_id: str, extension: str) -> str:
        return f"jobs/{meeting_date.isoformat()}/{job_id}/raw/input.{extension}"

    @staticmethod
    def raw_transcript_key(meeting_date: date, job_id: str) -> str:
        return f"jobs/{meeting_date.isoformat()}/{job_id}/raw/transcript.txt"

    @staticmethod
    def _curated_extraction_key(meeting_date: date, job_id: str) -> str:
        return f"jobs/{meeting_date.isoformat()}/{job_id}/curated/extraction.json"
