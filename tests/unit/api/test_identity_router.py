from __future__ import annotations

from unittest.mock import MagicMock

from seshat.app.platform.api.state import AppState
from seshat.core.models.enums import UserRole
from tests.unit.api.conftest import make_current_user


def _make_app_state() -> AppState:
    return AppState(
        config=MagicMock(),
        admin_service=MagicMock(),
        health_service=MagicMock(),
        graph_service=MagicMock(),
        job_service=MagicMock(),
    )


class TestMe:
    async def test_returns_user_identity(self, api_client):
        user = make_current_user(user_id="alice", role=UserRole.OPERATOR)
        async with api_client(_make_app_state(), user) as ac:
            resp = await ac.get("/me")
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "alice"
        assert resp.json()["role"] == "operator"

    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/me")
        assert resp.status_code == 401
