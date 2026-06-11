import mlflow


def log_token_metrics(
    stage: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    """Log LLM token counts (including cache) as MLflow metrics to the active run.

    No-ops when no run is active so it is safe to call from the production pipeline
    before MLflow is wired there.
    """
    if not mlflow.active_run():
        return
    mlflow.log_metrics(
        {
            f"{stage}.llm_input": float(input_tokens),
            f"{stage}.llm_output": float(output_tokens),
            f"{stage}.cache_read_input_tokens": float(cache_read_tokens),
            f"{stage}.cache_creation_input_tokens": float(cache_creation_tokens),
        }
    )
