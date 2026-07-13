from __future__ import annotations

import bcrypt
import pytest

from seshat.app.platform.api.auth import AuthenticationError, verify_api_key


class TestVerifyApiKey:
    def _hash(self, key: str) -> str:
        return bcrypt.hashpw(key.encode(), bcrypt.gensalt(rounds=4)).decode()

    async def test_valid_key_returns_user(self):
        key = "test-key-abc123"
        key_hash = self._hash(key)
        user_id, role = await verify_api_key(key, [(key_hash, "alice", "reviewer")])
        assert user_id == "alice"
        assert role == "reviewer"

    async def test_invalid_key_raises(self):
        key = "test-key-abc123"
        key_hash = self._hash("different-key")
        with pytest.raises(AuthenticationError):
            await verify_api_key(key, [(key_hash, "alice", "reviewer")])

    async def test_empty_store_raises(self):
        with pytest.raises(AuthenticationError):
            await verify_api_key("any-key", [])

    async def test_corrupt_hash_raises_authentication_error(self):
        with pytest.raises(AuthenticationError):
            await verify_api_key("any-key", [("not-a-valid-bcrypt-hash", "alice", "viewer")])

    async def test_corrupt_hash_before_valid_row_still_matches(self):
        valid_hash = self._hash("good-key")
        user_id, role = await verify_api_key(
            "good-key",
            [("not-a-valid-hash", "bob", "admin"), (valid_hash, "alice", "reviewer")],
        )
        assert user_id == "alice"
        assert role == "reviewer"

    async def test_corrupt_hash_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING), pytest.raises(AuthenticationError):
            await verify_api_key("any-key", [("not-a-valid-bcrypt-hash", "alice", "viewer")])
        assert any("alice" in r.message for r in caplog.records)
