from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from seshat.eval.mlflow_logging import (
    _log_breakdown_artifact,
    _set_active_model_with_params,
    configure_trace_processors,
    log_eval_run_metadata,
    log_retrieval_model,
    make_input_redactor,
)


class TestSetActiveModelWithParams:
    def test_logs_params_on_first_call(self):
        model_info = MagicMock()
        model_info.params = {}
        model_info.model_id = "model-abc"

        with (
            patch("seshat.eval.mlflow_logging.mlflow.set_active_model", return_value=model_info),
            patch("seshat.eval.mlflow_logging.mlflow.log_model_params") as mock_log,
        ):
            result = _set_active_model_with_params("mymodel", {"a": "1"})

        assert result is model_info
        mock_log.assert_called_once_with({"a": "1"})

    def test_skips_params_when_already_logged(self):
        model_info = MagicMock()
        model_info.params = {"a": "1"}

        with (
            patch("seshat.eval.mlflow_logging.mlflow.set_active_model", return_value=model_info),
            patch("seshat.eval.mlflow_logging.mlflow.log_model_params") as mock_log,
        ):
            _set_active_model_with_params("mymodel", {"a": "1"})

        mock_log.assert_not_called()

    def test_versioned_name_embeds_fingerprint(self):
        model_info = MagicMock()
        model_info.params = {}

        with (
            patch("seshat.eval.mlflow_logging.mlflow.set_active_model", return_value=model_info) as mock_set,
            patch("seshat.eval.mlflow_logging.mlflow.log_model_params"),
        ):
            _set_active_model_with_params("agent", {"p": "v"})

        called_name = mock_set.call_args[1]["name"]
        assert called_name.startswith("agent-")
        assert len(called_name) > len("agent-")


class TestLogRetrievalModel:
    def test_returns_model_id(self):
        model_info = MagicMock()
        model_info.params = {}
        model_info.model_id = "retrieval-001"

        from seshat.core.config.settings import VectorIndexConfig

        config = VectorIndexConfig()
        with (
            patch("seshat.eval.mlflow_logging.mlflow.set_active_model", return_value=model_info),
            patch("seshat.eval.mlflow_logging.mlflow.log_model_params"),
        ):
            result = log_retrieval_model("retrieval", config)

        assert result == "retrieval-001"


class TestLogEvalRunMetadata:
    def test_logs_params_metrics_and_tags(self):
        ex1, ex2 = MagicMock(), MagicMock()
        ex1.tags = {}
        ex2.tags = {}
        with (
            patch("seshat.eval.mlflow_logging.mlflow.log_params") as mock_params,
            patch("seshat.eval.mlflow_logging.mlflow.log_metrics") as mock_metrics,
            patch("seshat.eval.mlflow_logging.mlflow.set_tags") as mock_tags,
        ):
            log_eval_run_metadata(
                run_id="run-1",
                harness="identification",
                gate_passed=True,
                corpus_dir=Path("/data/corpus"),
                corpus_examples=[ex1, ex2],
            )

        mock_params.assert_called_once()
        params_arg = mock_params.call_args[0][0]
        assert params_arg["corpus.size"] == "2"

        mock_metrics.assert_called_once_with({"gate.passed": 1.0}, run_id="run-1")

        mock_tags.assert_called_once()
        tags_arg = mock_tags.call_args[0][0]
        assert tags_arg["harness"] == "identification"
        assert tags_arg["gate.passed"] == "true"

    def test_cache_hits_included_when_provided(self):
        with (
            patch("seshat.eval.mlflow_logging.mlflow.log_params"),
            patch("seshat.eval.mlflow_logging.mlflow.log_metrics"),
            patch("seshat.eval.mlflow_logging.mlflow.set_tags") as mock_tags,
        ):
            log_eval_run_metadata(
                run_id="run-2",
                harness="resolution",
                gate_passed=False,
                corpus_dir=Path("/data"),
                corpus_examples=[],  # empty — no tags iteration
                cache_hits=5,
                total_predictions=10,
            )

        tags_arg = mock_tags.call_args[0][0]
        assert tags_arg["cached"] == "false"

    def test_breakdown_artifact_logged_when_provided(self):
        with (
            patch("seshat.eval.mlflow_logging.mlflow.log_params"),
            patch("seshat.eval.mlflow_logging.mlflow.log_metrics"),
            patch("seshat.eval.mlflow_logging.mlflow.set_tags"),
            patch("seshat.eval.mlflow_logging._log_breakdown_artifact") as mock_breakdown,
        ):
            log_eval_run_metadata(
                run_id="run-3",
                harness="grouping",
                gate_passed=True,
                corpus_dir=Path("/data"),
                corpus_examples=[],
                breakdown_artifact={"key": "val"},
            )

        mock_breakdown.assert_called_once_with({"key": "val"}, "run-3")


class TestLogBreakdownArtifact:
    def test_writes_json_and_logs_artifact(self, tmp_path):
        with patch("seshat.eval.mlflow_logging.mlflow.log_artifact") as mock_log:
            _log_breakdown_artifact({"score": 0.9}, "run-x")

        mock_log.assert_called_once()
        kwargs = mock_log.call_args[1]
        assert kwargs["artifact_path"] == "eval"
        assert kwargs["run_id"] == "run-x"


class TestConfigureTraceProcessors:
    def test_passes_processors_to_mlflow(self):
        proc = MagicMock()
        with patch("seshat.eval.mlflow_logging.mlflow.tracing.configure") as mock_cfg:
            configure_trace_processors(proc)
        mock_cfg.assert_called_once_with(span_processors=[proc])


class TestMakeInputRedactor:
    def test_redacts_specified_fields(self):
        processor = make_input_redactor(fields_to_redact={"transcript"})
        span = MagicMock()
        span.inputs = {"transcript": "secret text", "query": "public"}
        processor(span)
        span.set_inputs.assert_called_once_with({"transcript": "**[REDACTED]**", "query": "public"})

    def test_excludes_specified_fields(self):
        processor = make_input_redactor(fields_to_exclude={"internal_key"})
        span = MagicMock()
        span.inputs = {"internal_key": "value", "query": "public"}
        processor(span)
        span.set_inputs.assert_called_once_with({"query": "public"})

    def test_no_change_does_not_call_set_inputs(self):
        processor = make_input_redactor()
        span = MagicMock()
        span.inputs = {"query": "public"}
        processor(span)
        span.set_inputs.assert_not_called()

    def test_empty_inputs_skips_processing(self):
        processor = make_input_redactor(fields_to_redact={"transcript"})
        span = MagicMock()
        span.inputs = None
        processor(span)
        span.set_inputs.assert_not_called()
