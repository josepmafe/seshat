from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from seshat.app.platform.api.dependencies import get_app_state
from seshat.app.platform.api.routers.admin import _get_root_key, _require_root_key
from seshat.app.services.admin import ApiKeyAlreadyRevokedError, ApiKeyNotFoundError
from tests.unit.app.platform.api.conftest import make_app_state


def _make_app_state():
    admin_service = MagicMock()
    admin_service.create_api_key = AsyncMock(return_value=("plaintext-key", "alice", "reviewer"))
    admin_service.list_api_keys = AsyncMock(return_value=[])
    admin_service.revoke_api_key = AsyncMock()
    return make_app_state(admin_service=admin_service)


def _make_key_row(key_id: int = 1, *, revoked: bool = False) -> dict:
    now = datetime.now(UTC)
    return {
        "id": key_id,
        "user_id": "alice",
        "role": "reviewer",
        "created_at": now,
        "revoked_at": now if revoked else None,
    }


class TestCreateApiKey:
    async def test_missing_header_returns_401(self, app, api_client):
        app.dependency_overrides[_get_root_key] = lambda: "correct-secret"
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/admin/api-keys", json={"user_id": "bob", "role": "reviewer"})
        assert resp.status_code == 401

    async def test_wrong_key_returns_401(self, app, api_client):
        app.dependency_overrides[_get_root_key] = lambda: "correct-secret"
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post(
                "/admin/api-keys",
                json={"user_id": "bob", "role": "reviewer"},
                headers={"X-API-Key": "wrong-secret"},
            )
        assert resp.status_code == 401

    async def test_creates_key_and_returns_plaintext(self, app, api_client):
        state = _make_app_state()
        app.dependency_overrides[_get_root_key] = lambda: "correct-secret"
        async with api_client(state) as ac:
            resp = await ac.post(
                "/admin/api-keys",
                json={"user_id": "alice", "role": "reviewer"},
                headers={"X-API-Key": "correct-secret"},
            )
        assert resp.status_code == 201
        body = resp.json()
        assert "api_key" in body
        assert body["user_id"] == "alice"
        assert body["role"] == "reviewer"
        state.admin_service.create_api_key.assert_called_once()

    async def test_empty_user_id_returns_422(self, app, api_client):
        app.dependency_overrides[_get_root_key] = lambda: "correct-secret"
        state = _make_app_state()
        async with api_client(state) as ac:
            resp = await ac.post(
                "/admin/api-keys",
                json={"user_id": "", "role": "reviewer"},
                headers={"X-API-Key": "correct-secret"},
            )
        assert resp.status_code == 422
        state.admin_service.create_api_key.assert_not_called()

    async def test_whitespace_user_id_returns_422(self, app, api_client):
        app.dependency_overrides[_get_root_key] = lambda: "correct-secret"
        state = _make_app_state()
        async with api_client(state) as ac:
            resp = await ac.post(
                "/admin/api-keys",
                json={"user_id": "   ", "role": "reviewer"},
                headers={"X-API-Key": "correct-secret"},
            )
        assert resp.status_code == 422
        state.admin_service.create_api_key.assert_not_called()


class TestListApiKeys:
    async def test_returns_keys_with_is_active(self, app, api_client):
        state = _make_app_state()
        state.admin_service.list_api_keys = AsyncMock(return_value=[_make_key_row(1), _make_key_row(2, revoked=True)])
        app.dependency_overrides[_get_root_key] = lambda: "secret"
        async with api_client(state) as ac:
            resp = await ac.get("/admin/api-keys", headers={"X-API-Key": "secret"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["is_active"] is True
        assert body[1]["is_active"] is False

    async def test_wrong_key_returns_401(self, app, api_client):
        app.dependency_overrides[_get_root_key] = lambda: "secret"
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/admin/api-keys", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401


class TestRevokeApiKey:
    async def test_revokes_key_returns_204(self, app, api_client):
        state = _make_app_state()
        app.dependency_overrides[_get_root_key] = lambda: "secret"
        async with api_client(state) as ac:
            resp = await ac.delete("/admin/api-keys/1", headers={"X-API-Key": "secret"})
        assert resp.status_code == 204
        state.admin_service.revoke_api_key.assert_called_once()

    async def test_not_found_returns_404(self, app, api_client):
        state = _make_app_state()
        state.admin_service.revoke_api_key = AsyncMock(side_effect=ApiKeyNotFoundError(99))
        app.dependency_overrides[_get_root_key] = lambda: "secret"
        async with api_client(state) as ac:
            resp = await ac.delete("/admin/api-keys/99", headers={"X-API-Key": "secret"})
        assert resp.status_code == 404

    async def test_already_revoked_returns_409(self, app, api_client):
        state = _make_app_state()
        state.admin_service.revoke_api_key = AsyncMock(side_effect=ApiKeyAlreadyRevokedError(1))
        app.dependency_overrides[_get_root_key] = lambda: "secret"
        async with api_client(state) as ac:
            resp = await ac.delete("/admin/api-keys/1", headers={"X-API-Key": "secret"})
        assert resp.status_code == 409


class TestMissingRootKey:
    async def test_missing_env_var_returns_500(self, app):
        # _get_root_key is NOT overridden; resolver.get_secret raises KeyError → 500.
        # Pins current behaviour: a misconfigured deployment returns 500, not a client error.
        app.dependency_overrides[get_app_state] = lambda: _make_app_state()
        resolver = MagicMock()
        resolver.get_secret = MagicMock(side_effect=KeyError("ROOT_API_KEY_SECRET"))
        try:
            with patch("seshat.app.platform.api.routers.admin.get_secrets_resolver", return_value=resolver):
                async with AsyncClient(
                    transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test/v1"
                ) as ac:
                    resp = await ac.get("/admin/api-keys", headers={"X-API-Key": "any"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 500

    async def test_empty_env_var_returns_500(self, app):
        app.dependency_overrides[get_app_state] = lambda: _make_app_state()
        resolver = MagicMock()
        resolver.get_secret = MagicMock(side_effect=ValueError("Secret is set but empty"))
        try:
            with patch("seshat.app.platform.api.routers.admin.get_secrets_resolver", return_value=resolver):
                async with AsyncClient(
                    transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test/v1"
                ) as ac:
                    resp = await ac.get("/admin/api-keys", headers={"X-API-Key": "any"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 500


class TestNonAsciiApiKey:
    async def test_non_ascii_key_raises_401(self):
        # httpx cannot send non-ASCII header values at all, so we call _require_root_key
        # directly: compare_digest raises TypeError for non-ASCII → must map to 401, not 500.
        with pytest.raises(HTTPException) as exc_info:
            await _require_root_key(root_key="correct-secret", x_api_key="roté")
        assert exc_info.value.status_code == 401
