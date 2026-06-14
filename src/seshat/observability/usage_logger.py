import mlflow


def log_token_metrics(
    stage: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    embedding_input_tokens: int = 0,
    metrics_prefix: str = "usage.",
) -> None:
    """Log LLM and embedding token counts as MLflow metrics to the active run.

    No-ops when no run is active so it is safe to call from the production pipeline
    before MLflow is wired there.
    """
    if not mlflow.active_run():
        return

    if stage:
        stage = stage.replace(".", "_").replace(" ", "_").replace("-", "_")  # sanitize stage name for metric keys
        metrics_prefix = f"{metrics_prefix}{stage}."

    mlflow.log_metrics(
        {
            f"{metrics_prefix}llm_input": float(input_tokens),
            f"{metrics_prefix}llm_output": float(output_tokens),
            f"{metrics_prefix}cache_read_input_tokens": float(cache_read_tokens),
            f"{metrics_prefix}cache_creation_input_tokens": float(cache_creation_tokens),
            f"{metrics_prefix}embedding_input": float(embedding_input_tokens),
        }
    )
