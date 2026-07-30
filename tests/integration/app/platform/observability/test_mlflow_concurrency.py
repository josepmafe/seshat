"""Integration tests proving the concurrency-safety claim behind GraphService's shared
search MLflow run: mlflow.start_span(run_id=...) and MlflowClient calls don't corrupt each
other's state when multiple search() calls run concurrently on the same event loop, unlike
mlflow.start_run() (thread-local active-run stack, unsafe for concurrent async callers).
"""

from __future__ import annotations

import asyncio

import mlflow
import pytest
from mlflow import MlflowClient

from seshat.app.platform.observability.mlflow_run_logging import _log_metrics_to_run

pytestmark = pytest.mark.integration


@pytest.fixture
def mlflow_run(isolated_mlflow_tracking):
    experiment_id = mlflow.create_experiment("concurrency-test")
    client = MlflowClient()
    run = client.create_run(experiment_id)
    return client, run.info.run_id, experiment_id


class TestConcurrentSpans:
    async def test_concurrent_start_span_calls_each_produce_an_independent_trace(self, mlflow_run):
        _client, run_id, experiment_id = mlflow_run

        async def _traced_call(i: int) -> int:
            with mlflow.start_span(name="graph_search", run_id=run_id) as span:
                await asyncio.sleep(0.01)
                span.set_attribute("call_index", i)
            return i

        results = await asyncio.gather(*[_traced_call(i) for i in range(10)])
        mlflow.flush_trace_async_logging()

        assert sorted(results) == list(range(10))
        # Verified experimentally: mlflow.start_span(run_id=...) without an active experiment
        # or explicit trace_destination stores the trace under the "Default" experiment ("0"),
        # NOT the run's own experiment (here, experiment_id) — despite start_span's docstring
        # claiming otherwise. See the TODO on GraphService.search referencing this.
        traces = mlflow.search_traces(run_id=run_id, locations=[experiment_id, "0"])
        assert len(traces) == 10


class TestConcurrentMetricLogging:
    async def test_concurrent_log_metrics_to_run_calls_all_persist(self, mlflow_run):
        client, run_id, _experiment_id = mlflow_run

        async def _log_call(i: int) -> None:
            await asyncio.sleep(0.01)
            _log_metrics_to_run(run_id, {f"usage.call_{i}.llm_input": float(i)})

        await asyncio.gather(*[_log_call(i) for i in range(10)])
        mlflow.flush_async_logging()

        run = client.get_run(run_id)
        for i in range(10):
            assert run.data.metrics[f"usage.call_{i}.llm_input"] == float(i)
