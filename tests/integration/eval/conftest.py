import mlflow
import pytest


@pytest.fixture(autouse=True)
def isolate_mlflow_run():
    """End any active MLflow run after each eval test.

    mlflow.genai.evaluate() can leave an active run open in the process-global MLflow
    context. Without this, subsequent tests in the same session reuse that run and
    mlflow.log_params() rejects the conflicting param values.
    """
    yield
    mlflow.end_run()
