from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.blob_store.s3_store import S3BlobStore

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig


def get_blob_store(config: SeshatConfig) -> S3BlobStore:
    return S3BlobStore(config.blob_store)
