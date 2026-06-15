from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from seshat.blob_store.s3_store import S3BlobStore

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig


@pytest.fixture
def store(minimal_config: SeshatConfig) -> S3BlobStore:
    return S3BlobStore(minimal_config.blob_store)


class TestBlobPathsMixin:
    def test_raw_input_key(self, store: S3BlobStore):
        key = store.raw_input_key(date(2026, 4, 21), "job-1", "mp3")
        assert key == "jobs/2026-04-21/job-1/raw/input.mp3"

    def test_raw_transcript_key(self, store: S3BlobStore):
        key = store.raw_transcript_key(date(2026, 4, 21), "job-1")
        assert key == "jobs/2026-04-21/job-1/raw/transcript.txt"

    def test_curated_extraction_key(self, store: S3BlobStore):
        key = store.curated_extraction_key(date(2026, 4, 21), "job-1")
        assert key == "jobs/2026-04-21/job-1/curated/extraction.json"

    def test_init_source_key(self, store: S3BlobStore):
        key = store.init_source_key("init-1", 3)
        assert key == "init/init-1/source/0003.md"

    def test_init_curated_extraction_key(self, store: S3BlobStore):
        key = store.init_curated_extraction_key("init-1")
        assert key == "init/init-1/curated/extraction.json"
