from pathlib import Path

import pytest

from seshat.config.settings import EvalConfig, ObservabilityConfig
from seshat.eval.models import GateResult
from tests.integration.conftest import SKIP_IF_NO_LLM_API
from tests.integration.eval.helpers import make_resolution_runner

CORPUS_BASE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "eval" / "test_corpus"

pytestmark = [pytest.mark.integration, pytest.mark.llm, SKIP_IF_NO_LLM_API]


class TestResolutionEvalRunner:
    async def test_run_produces_gate_result_with_resolution_metrics(self, tmp_path):
        config = EvalConfig(
            corpus_base_dir=CORPUS_BASE_DIR,
            gate_path=tmp_path / "eval_gate.json",
            observability=ObservabilityConfig(
                mlflow_tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
                mlflow_experiment_name="seshat-resolution-eval-test",
            ),
        )
        runner = make_resolution_runner(config)
        result = await runner.run()

        assert isinstance(result, GateResult)
        assert result.run_id
        assert result.resolution_metrics is not None
        assert "precision" in result.resolution_metrics
        assert "recall" in result.resolution_metrics
        assert (tmp_path / "eval_gate.json").exists()
