from __future__ import annotations

import pytest

from seshat.blob_store.s3_store import S3BlobStore
from seshat.core.config.settings import BlobStoreConfig
from tests.integration.conftest import LOCALSTACK_REGION, LOCALSTACK_TEST_BUCKET


@pytest.fixture
async def blob_store(localstack_s3_url):
    """Override shared blob_store fixture — S3BlobStore directly for low-level store tests."""
    config = BlobStoreConfig(
        bucket=LOCALSTACK_TEST_BUCKET,
        region=LOCALSTACK_REGION,
        endpoint_url=localstack_s3_url,
    )
    store = S3BlobStore(config)
    await store.connect()
    yield store
    await store.close()
