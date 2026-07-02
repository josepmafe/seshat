from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

from seshat.repositories.blob_repository import BlobRepository


def _make_repo(**store_returns) -> tuple[BlobRepository, MagicMock]:
    store = MagicMock()
    store.put = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=None)
    for method, return_value in store_returns.items():
        setattr(store, method, AsyncMock(return_value=return_value))
    return BlobRepository(store), store


class TestPutByKey:
    async def test_delegates_put(self):
        repo, store = _make_repo()
        await repo.put_by_key("some/key", b"data")
        store.put.assert_awaited_once_with("some/key", b"data")


class TestGetByKey:
    async def test_returns_bytes_when_found(self):
        repo, _store = _make_repo(get=b"hello")
        result = await repo.get_by_key("some/key")
        assert result == b"hello"

    async def test_returns_none_when_missing(self):
        repo, _store = _make_repo(get=None)
        result = await repo.get_by_key("missing/key")
        assert result is None


class TestKeyConventions:
    def test_raw_input_key(self):
        key = BlobRepository.raw_input_key(date(2026, 6, 1), "job-abc", "txt")
        assert key == "jobs/2026-06-01/job-abc/raw/input.txt"

    def test_raw_transcript_key(self):
        key = BlobRepository.raw_transcript_key(date(2026, 6, 1), "job-abc")
        assert key == "jobs/2026-06-01/job-abc/raw/transcript.txt"

    def test_curated_extraction_key(self):
        key = BlobRepository._curated_extraction_key(date(2026, 6, 1), "job-abc")
        assert key == "jobs/2026-06-01/job-abc/curated/extraction.json"

    def test_raw_input_key_extension_preserved(self):
        key = BlobRepository.raw_input_key(date(2026, 6, 1), "job-abc", "mp3")
        assert key.endswith(".mp3")


class TestPutRawInput:
    async def test_writes_to_correct_key(self):
        repo, store = _make_repo()
        await repo.put_raw_input(date(2026, 6, 1), "job-abc", "txt", b"content")
        expected_key = "jobs/2026-06-01/job-abc/raw/input.txt"
        store.put.assert_awaited_once_with(expected_key, b"content")


class TestPutCuratedExtraction:
    async def test_writes_to_correct_key(self):
        repo, store = _make_repo()
        await repo.put_curated_extraction(date(2026, 6, 1), "job-abc", b'{"nodes":[]}')
        expected_key = "jobs/2026-06-01/job-abc/curated/extraction.json"
        store.put.assert_awaited_once_with(expected_key, b'{"nodes":[]}')

    async def test_get_returns_bytes(self):
        repo, _store = _make_repo(get=b'{"nodes":[]}')
        result = await repo.get_curated_extraction(date(2026, 6, 1), "job-abc")
        assert result == b'{"nodes":[]}'


class TestPutRawTranscript:
    async def test_writes_to_correct_key(self):
        repo, store = _make_repo()
        await repo.put_raw_transcript(date(2026, 6, 1), "job-abc", b"transcript text")
        expected_key = "jobs/2026-06-01/job-abc/raw/transcript.txt"
        store.put.assert_awaited_once_with(expected_key, b"transcript text")


class TestGetRawInput:
    async def test_delegates_get_with_correct_key(self):
        repo, store = _make_repo(get=b"audio bytes")
        result = await repo.get_raw_input(date(2026, 6, 1), "job-abc", "mp3")
        expected_key = "jobs/2026-06-01/job-abc/raw/input.mp3"
        store.get.assert_awaited_once_with(expected_key)
        assert result == b"audio bytes"

    async def test_returns_none_when_missing(self):
        repo, _store = _make_repo(get=None)
        result = await repo.get_raw_input(date(2026, 6, 1), "job-abc", "mp3")
        assert result is None


class TestGetRawTranscript:
    async def test_delegates_get_with_correct_key(self):
        repo, store = _make_repo(get=b"transcript")
        result = await repo.get_raw_transcript(date(2026, 6, 1), "job-abc")
        expected_key = "jobs/2026-06-01/job-abc/raw/transcript.txt"
        store.get.assert_awaited_once_with(expected_key)
        assert result == b"transcript"
