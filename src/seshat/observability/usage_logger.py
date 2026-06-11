import mlflow


def log_usage(stage: str, input_tokens: int, output_tokens: int) -> None:
    """Log LLM token counts as MLflow metrics to the active run.

    No-ops when no run is active so it is safe to call from the production pipeline
    before MLflow is wired there.
    """
    if not mlflow.active_run():
        return
    mlflow.log_metrics(
        {
            f"{stage}.llm_input": float(input_tokens),
            f"{stage}.llm_output": float(output_tokens),
        }
    )


def log_cache_metrics(
    stage: str,
    cache_read_tokens: int,
    cache_write_tokens: int,
    cached_tokens: int | None = None,
) -> None:
    """Log prompt cache hit/miss metrics to the active run.

    No-ops when no run is active so it is safe to call from the production pipeline
    before MLflow is wired there.
    """
    if not mlflow.active_run():
        return
    metrics = {
        f"{stage}.cache_read_input_tokens": float(cache_read_tokens),
        f"{stage}.cache_creation_input_tokens": float(cache_write_tokens),
    }
    if cached_tokens is not None:
        metrics[f"{stage}.cached_tokens"] = float(cached_tokens)
    mlflow.log_metrics(metrics)
