from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError

from seshat.infra.blob_store.s3_store import S3BlobStore, _s3_should_retry

if TYPE_CHECKING:
    from seshat.core.config.settings import SeshatConfig


@pytest.fixture
def store(minimal_config: SeshatConfig) -> S3BlobStore:
    return S3BlobStore(minimal_config.blob_store)


def _make_mock_session():
    """Return a mock aiobotocore session whose .create_client() is an async context manager."""
    fake_client = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__.return_value = fake_client
    ctx.__aexit__.return_value = None
    session = MagicMock()
    session.create_client.return_value = ctx
    return session, ctx, fake_client


class TestS3BlobStoreLifecycle:
    async def test_connect_sets_client(self, store: S3BlobStore):
        session, ctx, fake_client = _make_mock_session()
        store._session = session

        await store.connect()

        assert store._client is fake_client
        assert store._client_ctx is ctx
        ctx.__aenter__.assert_awaited_once()

    async def test_close_clears_client(self, store: S3BlobStore):
        session, ctx, _ = _make_mock_session()
        store._session = session
        await store.connect()

        await store.close()

        assert store._client is None
        assert store._client_ctx is None
        ctx.__aexit__.assert_awaited_once_with(None, None, None)

    async def test_close_is_idempotent(self, store: S3BlobStore):
        """close() on a not-yet-connected store must not raise."""
        await store.close()

    async def test_client_property_raises_before_connect(self, store: S3BlobStore):
        with pytest.raises(RuntimeError, match="connect"):
            _ = store.client


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "test"}}, "GetObject")


class TestS3BlobStoreGet:
    def _store_with_client(self, store: S3BlobStore) -> tuple[S3BlobStore, MagicMock]:
        fake_client = MagicMock()
        store._client = fake_client
        return store, fake_client

    async def test_returns_bytes_on_success(self, store: S3BlobStore):
        store, client = self._store_with_client(store)
        body = AsyncMock()
        body.read = AsyncMock(return_value=b"hello")
        client.get_object = AsyncMock(return_value={"Body": body})

        result = await store.get("some/key")

        assert result == b"hello"

    async def test_returns_none_on_no_such_key(self, store: S3BlobStore):
        store, client = self._store_with_client(store)
        client.get_object = AsyncMock(side_effect=_client_error("NoSuchKey"))

        result = await store.get("missing/key")

        assert result is None

    async def test_returns_none_on_404(self, store: S3BlobStore):
        store, client = self._store_with_client(store)
        client.get_object = AsyncMock(side_effect=_client_error("404"))

        result = await store.get("missing/key")

        assert result is None

    async def test_reraises_on_403(self, store: S3BlobStore):
        store, client = self._store_with_client(store)
        client.get_object = AsyncMock(side_effect=_client_error("403"))

        with pytest.raises(ClientError):
            await store.get("forbidden/key")


class TestS3ShouldRetry:
    def test_returns_false_for_404(self):
        assert _s3_should_retry(_client_error("404")) is False

    def test_returns_false_for_403(self):
        assert _s3_should_retry(_client_error("403")) is False

    def test_returns_false_for_no_such_key(self):
        assert _s3_should_retry(_client_error("NoSuchKey")) is False

    def test_returns_false_for_no_such_bucket(self):
        assert _s3_should_retry(_client_error("NoSuchBucket")) is False

    def test_returns_false_for_400(self):
        assert _s3_should_retry(_client_error("400")) is False

    def test_returns_true_for_500(self):
        assert _s3_should_retry(_client_error("500")) is True

    def test_returns_true_for_slow_down(self):
        assert _s3_should_retry(_client_error("SlowDown")) is True

    def test_returns_true_for_non_client_error(self):
        assert _s3_should_retry(RuntimeError("network")) is True
