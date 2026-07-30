"""Integration tests for _get_search_run_id against a real (disposable, local) MLflow store."""

from __future__ import annotations

import mlflow
import pytest
from mlflow import MlflowClient

from seshat.app.platform.api.state import _get_search_run_id
from seshat.core.config.settings import ObservabilityConfig

pytestmark = pytest.mark.integration

_EXPERIMENT_NAME = "search-run-test"


@pytest.fixture
def observability_config(isolated_mlflow_tracking) -> ObservabilityConfig:
    mlflow.create_experiment(_EXPERIMENT_NAME)
    return ObservabilityConfig(mlflow_experiment_name=_EXPERIMENT_NAME)


class TestGetSearchRunId:
    def test_creates_run_under_configured_experiment(self, observability_config):
        client = MlflowClient()

        run_id = _get_search_run_id(client, observability_config)

        run = client.get_run(run_id)
        experiment = mlflow.get_experiment_by_name(_EXPERIMENT_NAME)
        assert run.info.experiment_id == experiment.experiment_id

    def test_run_name_has_search_run_prefix(self, observability_config):
        client = MlflowClient()

        run_id = _get_search_run_id(client, observability_config)

        run = client.get_run(run_id)
        assert run.info.run_name.startswith("search-run-")

    def test_run_tagged_with_graph_search_source(self, observability_config):
        client = MlflowClient()

        run_id = _get_search_run_id(client, observability_config)

        run = client.get_run(run_id)
        assert run.data.tags["source"] == "graph_search"

    def test_raises_when_experiment_does_not_exist(self, isolated_mlflow_tracking):
        # setup_mlflow() (which calls mlflow.set_experiment) is expected to have run before
        # build_app_state — this asserts the contract is actually enforced, not silently
        # skipped, if that ordering is ever broken. No experiment created here on purpose.
        client = MlflowClient()
        config = ObservabilityConfig(mlflow_experiment_name=_EXPERIMENT_NAME)

        with pytest.raises(AssertionError):
            _get_search_run_id(client, config)
