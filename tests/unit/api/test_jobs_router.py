from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from seshat.api.app import create_app
from seshat.api.dependencies import _get_current_user, get_app_state
from seshat.api.state import AppState
from seshat.models.enums import NodeStatus, UserRole
from seshat.models.nodes import ExtractionResult
from tests.helpers import make_node


def _make_app_state(**overrides) -> AppState:
    ops = MagicMock()
    ops.find_job_by_idempotency_key = AsyncMock(return_value=None)
    ops.count_recent_jobs_for_user = AsyncMock(return_value=0)
    ops.count_running_jobs = AsyncMock(return_value=0)
    ops.create_job = AsyncMock()
    ops.get_job = AsyncMock(return_value=None)
    ops.reset_failed_job = AsyncMock()
    ops.fail_job = AsyncMock()

    config = MagicMock()
    config.max_jobs_per_user_per_hour = 10
    config.max_concurrent_jobs = 5

    queue = MagicMock()
    queue.enqueue = AsyncMock()

    runner = MagicMock()

    state = AppState(
        ops=ops,
        kb_store=MagicMock(),
        config=config,
        queue=queue,
        results={},
        runner=runner,
    )
    for k, v in overrides.items():
        object.__setattr__(state, k, v)
    return state


def _make_current_user(user_id: str = "alice", role: UserRole = UserRole.OPERATOR):
    from seshat.api.dependencies import CurrentUser

    return CurrentUser(user_id=user_id, role=role)


def _make_job_row(status: str = "pending") -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "status": status,
        "idempotency_key": None,
        "created_at": datetime.now(UTC),
        "error_payload": None,
        "mlflow_run_id": None,
    }


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test/v1")


def _override(app, state: AppState, user=None):
    app.dependency_overrides[get_app_state] = lambda: state
    if user is not None:
        app.dependency_overrides[_get_current_user] = lambda: user


def _clear(app):
    app.dependency_overrides.clear()


class TestSubmitJob:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": "{}"})
        _clear(app)
        assert resp.status_code == 401

    async def test_returns_job_id(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user())
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with client as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        _clear(app)
        assert resp.status_code == 202
        assert "job_id" in resp.json()

    async def test_idempotency_returns_existing_job(self, app, client):
        state = _make_app_state()
        state.ops.find_job_by_idempotency_key = AsyncMock(return_value={"job_id": "existing-job", "status": "pending"})
        _override(app, state, _make_current_user())
        body = json.dumps(
            {"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}, "idempotency_key": "key-abc"}
        )
        async with client as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        _clear(app)
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "existing-job"
        state.ops.create_job.assert_not_called()

    async def test_rate_limit_per_user(self, app, client):
        state = _make_app_state()
        state.ops.count_recent_jobs_for_user = AsyncMock(return_value=10)
        _override(app, state, _make_current_user())
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with client as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        _clear(app)
        assert resp.status_code == 429
        assert resp.json()["limit_type"] == "per_user_hourly_cap"

    async def test_rate_limit_global_concurrency(self, app, client):
        state = _make_app_state()
        state.ops.count_running_jobs = AsyncMock(return_value=5)
        _override(app, state, _make_current_user())
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with client as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        _clear(app)
        assert resp.status_code == 429
        assert resp.json()["limit_type"] == "global_concurrency_cap"

    async def test_viewer_cannot_submit(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.VIEWER))
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with client as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        _clear(app)
        assert resp.status_code == 403

    async def test_overrides_require_operator(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user(role=UserRole.REVIEWER))
        body = json.dumps(
            {"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}, "overrides": {"extraction": {}}}
        )
        async with client as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        _clear(app)
        assert resp.status_code == 403


class TestGetJob:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.get("/jobs/job-1")
        _clear(app)
        assert resp.status_code == 401

    async def test_not_found(self, app, client):
        state = _make_app_state()
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/jobs/job-1")
        _clear(app)
        assert resp.status_code == 404

    async def test_returns_job_response(self, app, client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("pending"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/jobs/job-1")
        _clear(app)
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-1"
        assert resp.json()["status"] == "pending"


class TestGetJobResults:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.get("/jobs/job-1/results")
        _clear(app)
        assert resp.status_code == 401

    async def test_results_not_ready(self, app, client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("pending"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/jobs/job-1/results")
        _clear(app)
        assert resp.status_code == 409

    async def test_returns_result_when_awaiting_review(self, app, client):
        node = make_node()
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.get("/jobs/job-1/results")
        _clear(app)
        assert resp.status_code == 200


class TestApproveJob:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        _clear(app)
        assert resp.status_code == 401

    async def test_requires_reviewer_or_operator(self, app, client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        _override(app, state, _make_current_user(role=UserRole.VIEWER))
        async with client as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        _clear(app)
        assert resp.status_code == 403

    async def test_not_awaiting_review(self, app, client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("pending"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        _clear(app)
        assert resp.status_code == 409

    def _result_nodes(self, state: AppState) -> dict:
        return {str(n.id): n for n in state.results["job-1"].nodes}

    async def test_bulk_rule_approves_above_threshold(self, app, client):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"approve_above_threshold": {"threshold": 0.8}})
        _clear(app)
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.APPROVED

    async def test_bulk_rule_skips_excluded_nodes(self, app, client):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"approve_above_threshold": {"threshold": 0.8, "exclude": [str(node.id)]}},
            )
        _clear(app)
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.PENDING_REVIEW

    async def test_individual_decision_approves(self, app, client):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": str(node.id), "action": "approve"}]},
            )
        _clear(app)
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.APPROVED

    async def test_individual_decision_rejects(self, app, client):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": str(node.id), "action": "reject"}]},
            )
        _clear(app)
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.REJECTED

    async def test_unknown_node_in_decisions_is_ignored(self, app, client):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": "00000000-0000-0000-0000-000000000000", "action": "approve"}]},
            )
        _clear(app)
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.PENDING_REVIEW


class TestRetryJob:
    async def test_requires_auth(self, app, client):
        _override(app, _make_app_state())
        async with client as ac:
            resp = await ac.post("/jobs/job-1/retry")
        _clear(app)
        assert resp.status_code == 401

    async def test_requires_operator(self, app, client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("failed"))
        _override(app, state, _make_current_user(role=UserRole.REVIEWER))
        async with client as ac:
            resp = await ac.post("/jobs/job-1/retry")
        _clear(app)
        assert resp.status_code == 403

    async def test_not_failed(self, app, client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("pending"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post("/jobs/job-1/retry")
        _clear(app)
        assert resp.status_code == 409

    async def test_resets_failed_job(self, app, client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("failed"))
        _override(app, state, _make_current_user())
        async with client as ac:
            resp = await ac.post("/jobs/job-1/retry")
        _clear(app)
        assert resp.status_code == 200
        state.ops.reset_failed_job.assert_called_once_with("job-1")
