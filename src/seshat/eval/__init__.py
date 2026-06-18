def require_eval_deps() -> None:
    try:
        import rapidfuzz  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The seshat.eval package requires optional dependencies that are not installed. Run: `uv sync --group eval`"
        ) from exc
