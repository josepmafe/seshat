import logging

import mlflow

from seshat.core.config.settings import ObservabilityConfig

logger = logging.getLogger(__name__)


def setup_mlflow(config: ObservabilityConfig, *, disable_autolog: bool = False) -> str:
    """Configure MLflow tracking, enable LangChain autolog, and resolve experiment ID.

    Returns the experiment ID. Called once at process startup. Autolog is kept on for eval
    too, so the agent LLM-call traces are captured in the MLflow UI; ``disable_autolog`` is
    retained as an escape hatch but no caller currently sets it.
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.langchain.autolog(disable=disable_autolog)  # type: ignore[attr-defined]

    experiment = mlflow.set_experiment(config.mlflow_experiment_name)
    logger.info(
        "MLflow configured: uri=%r, experiment=%r (id=%s)",
        config.mlflow_tracking_uri,
        config.mlflow_experiment_name,
        experiment.experiment_id,
    )
    return experiment.experiment_id


def mlflow_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    return f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
