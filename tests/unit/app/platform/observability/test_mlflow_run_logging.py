from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from seshat.app.platform.observability.mlflow_run_logging import (
    _log_metrics_to_run,
    log_identification_failures,
    log_resolution_failures,
    log_token_metrics,
    set_error_tag,
    set_phase_tag,
)
from seshat.core.models.enums import ConceptType
from seshat.core.models.nodes import FailedResolutionSource


def test_log_identification_failures_empty_list_does_not_log():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metric") as mock_log_metric,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag,
    ):
        log_identification_failures([])

    mock_log_metric.assert_not_called()
    mock_set_tag.assert_not_called()


def test_log_identification_failures_logs_count_and_tag():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metric") as mock_log_metric,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag,
    ):
        log_identification_failures([ConceptType.DECISION, ConceptType.RISK])

    mock_log_metric.assert_called_once_with("identification.failed_concept_types", 2)
    mock_set_tag.assert_called_once_with("identification.failed_concept_types", "decision,risk")


def test_log_resolution_failures_empty_list_does_not_log():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metric") as mock_log_metric,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag,
    ):
        log_resolution_failures([])

    mock_log_metric.assert_not_called()
    mock_set_tag.assert_not_called()


def test_log_resolution_failures_logs_count_and_tag():
    node_id = uuid4()
    failed = [FailedResolutionSource(node_id=node_id, concept_type=ConceptType.RISK)]
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metric") as mock_log_metric,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag,
    ):
        log_resolution_failures(failed)

    mock_log_metric.assert_called_once_with("resolution.failed_sources", 1)
    mock_set_tag.assert_called_once_with("resolution.failed_sources", str(node_id))


def test_log_token_metrics_no_active_run_does_not_call_log_metrics():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run", return_value=None),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("my_stage", input_tokens=10, output_tokens=5)

    mock_log.assert_not_called()


def test_log_token_metrics_with_active_run_logs_prefixed_keys():
    fake_run = MagicMock()
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run", return_value=fake_run),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics(
            "my_stage",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=3,
            cache_creation_tokens=2,
            embedding_input_tokens=7,
        )

    mock_log.assert_called_once_with(
        {
            "usage.my_stage.llm_input": 10.0,
            "usage.my_stage.llm_output": 5.0,
            "usage.my_stage.cache_read_input_tokens": 3.0,
            "usage.my_stage.cache_creation_input_tokens": 2.0,
            "usage.my_stage.embedding_input": 7.0,
        }
    )


def test_log_token_metrics_stage_sanitisation():
    fake_run = MagicMock()
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run", return_value=fake_run),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("step.one two-three", input_tokens=1, output_tokens=1)

    logged = mock_log.call_args[0][0]
    assert "usage.step_one_two_three.llm_input" in logged


def test_log_token_metrics_empty_stage_omits_stage_segment():
    fake_run = MagicMock()
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run", return_value=fake_run),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("", input_tokens=4, output_tokens=2)

    logged = mock_log.call_args[0][0]
    assert "usage.llm_input" in logged
    # No double-dot or spurious stage segment
    for key in logged:
        assert ".." not in key


def test_log_token_metrics_with_run_id_bypasses_active_run_check():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run") as mock_active_run,
        patch("seshat.app.platform.observability.mlflow_run_logging._log_metrics_to_run") as mock_log_to_run,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log_metrics,
    ):
        log_token_metrics("graph_search", input_tokens=10, output_tokens=5, run_id="run-123")

    mock_active_run.assert_not_called()
    mock_log_metrics.assert_not_called()
    mock_log_to_run.assert_called_once_with(
        "run-123",
        {"usage.graph_search.llm_input": 10.0, "usage.graph_search.llm_output": 5.0},
    )


def test_set_error_tag_truncates_to_250_chars():
    with patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag:
        set_error_tag(ValueError("x" * 300))

    logged_value = mock_set_tag.call_args[0][1]
    assert mock_set_tag.call_args[0][0] == "error"
    assert len(logged_value) == 250


def test_set_phase_tag_sets_phase_tag():
    with patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag:
        set_phase_tag("resolution")

    mock_set_tag.assert_called_once_with("phase", "resolution")


def test__log_metrics_to_run_logs_batch_with_explicit_run_id():
    mock_client = MagicMock()
    with patch("seshat.app.platform.observability.mlflow_run_logging.MlflowClient", return_value=mock_client):
        _log_metrics_to_run("run-123", {"usage.graph_search.llm_input": 10.0})

    mock_client.log_batch.assert_called_once()
    call_kwargs = mock_client.log_batch.call_args
    assert call_kwargs.args[0] == "run-123"
    logged_metrics = call_kwargs.kwargs["metrics"]
    assert len(logged_metrics) == 1
    assert logged_metrics[0].key == "usage.graph_search.llm_input"
    assert logged_metrics[0].value == 10.0
    assert logged_metrics[0].step == 0


def test__log_metrics_to_run_does_not_touch_active_run():
    """_log_metrics_to_run must never call mlflow.active_run() — that's the thread-local
    global this helper exists to bypass for concurrency safety."""
    mock_client = MagicMock()
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.MlflowClient", return_value=mock_client),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run") as mock_active_run,
    ):
        _log_metrics_to_run("run-123", {"a": 1.0})

    mock_active_run.assert_not_called()


def test__log_metrics_to_run_empty_metrics_calls_log_batch_with_empty_list():
    mock_client = MagicMock()
    with patch("seshat.app.platform.observability.mlflow_run_logging.MlflowClient", return_value=mock_client):
        _log_metrics_to_run("run-123", {})

    mock_client.log_batch.assert_called_once()
    assert mock_client.log_batch.call_args.kwargs["metrics"] == []


def test__log_metrics_to_run_empty_metrics_does_not_raise_against_real_mlflow():
    """log_batch tolerates an empty metrics list — verifies the real MLflow contract this
    helper relies on when log_token_metrics filters all-zero metrics down to {}."""
    import mlflow

    with mlflow.start_run() as run:
        _log_metrics_to_run(run.info.run_id, {})  # must not raise
