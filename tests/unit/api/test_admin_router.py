from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from seshat.api.routers.admin import _get_root_key
from seshat.api.state import AppState
from seshat.ops.ledger import ApiKeyAlreadyRevokedError, ApiKeyNotFoundError


def _make_app_state() -> AppState:
    ops = MagicMock()
    ops.create_api_key = AsyncMock()
    ops.list_api_keys = AsyncMock(return_value=[])
    ops.revoke_api_key = AsyncMock()
    return AppState(
        ops=ops,
        kb_store=MagicMock(),
        config=MagicMock(),
        queue=MagicMock(),
        runner=MagicMock(),
        manual_ingestion=MagicMock(),
        blob_store=MagicMock(),
    )


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
        state.ops.create_api_key.assert_called_once()


class TestListApiKeys:
    async def test_returns_keys_with_is_active(self, app, api_client):
        state = _make_app_state()
        state.ops.list_api_keys = AsyncMock(return_value=[_make_key_row(1), _make_key_row(2, revoked=True)])
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
        state.ops.revoke_api_key.assert_called_once()

    async def test_not_found_returns_404(self, app, api_client):
        state = _make_app_state()
        state.ops.revoke_api_key = AsyncMock(side_effect=ApiKeyNotFoundError(99))
        app.dependency_overrides[_get_root_key] = lambda: "secret"
        async with api_client(state) as ac:
            resp = await ac.delete("/admin/api-keys/99", headers={"X-API-Key": "secret"})
        assert resp.status_code == 404

    async def test_already_revoked_returns_409(self, app, api_client):
        state = _make_app_state()
        state.ops.revoke_api_key = AsyncMock(side_effect=ApiKeyAlreadyRevokedError(1))
        app.dependency_overrides[_get_root_key] = lambda: "secret"
        async with api_client(state) as ac:
            resp = await ac.delete("/admin/api-keys/1", headers={"X-API-Key": "secret"})
        assert resp.status_code == 409
