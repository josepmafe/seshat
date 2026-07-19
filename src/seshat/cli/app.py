from __future__ import annotations

import asyncio
import os
import selectors
import subprocess
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from seshat.eval.corpus_tags import CorpusTagFilter

import typer
from dotenv import load_dotenv

from seshat.app.platform.observability.mlflow_setup import setup_mlflow
from seshat.core.config.eval_settings import EvalConfig
from seshat.core.config.settings import GroundingLLMConfig, ObservabilityConfig, SeshatConfig
from seshat.core.utils.log import configure_logging, get_logger, set_job_id
from seshat.eval.mlflow_logging import configure_trace_processors

logger = get_logger(__name__)


def _run_async(coro: Coroutine) -> None:
    # psycopg (asyncpg-backed PGVector) is incompatible with Windows ProactorEventLoop
    if sys.platform == "win32":
        asyncio.run(coro, loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))
    else:
        asyncio.run(coro)


app = typer.Typer(name="seshat", help="Seshat — meeting knowledge base CLI", no_args_is_help=True)
eval_app = typer.Typer(help="Eval harnesses, calibration, and tooling", no_args_is_help=True)
app.add_typer(eval_app, name="eval")

_HARNESS_TYPES = ["grounding", "grouping", "identification", "resolution", "retrieval"]
_CALIBRATION_TYPES = ["retrieval", "identification"]


def _patch_httpx_ssl() -> None:
    import httpx

    _orig = httpx.Client.__init__

    def _no_verify(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("verify", False)
        _orig(self, *args, **kwargs)

    httpx.Client.__init__ = _no_verify  # type: ignore[method-assign]


@eval_app.command("harness")
def eval_cmd(
    harness: Annotated[
        str | None,
        typer.Argument(help=f"Harness to run: {' | '.join(_HARNESS_TYPES)}. Omit with --all to run all enabled."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Filter corpus by tag in `key=value` format. Repeatable."),
    ] = None,
    clear_cache: Annotated[
        bool,
        typer.Option("--clear-cache", help="Clear the prediction cache of each harness that runs, before running."),
    ] = False,
    run_all: Annotated[
        bool,
        typer.Option("--all", help="Run every harness whose EVAL__RUN_<harness> flag is enabled."),
    ] = False,
) -> None:
    """Run one evaluation harness, or every enabled harness with --all."""
    if harness is not None and run_all:
        typer.echo("Pass either a harness name or --all, not both.", err=True)
        raise typer.Exit(code=1)

    # Single named harness: the simple case — run it, and let any failure propagate (fail-hard).
    if harness is not None:
        if clear_cache:
            _clear_cache(harness)

        _run_single_harness(harness, tags)
        return

    if not run_all:
        typer.echo("Provide a harness name or --all.", err=True)
        raise typer.Exit(code=1)

    harnesses = EvalConfig().enabled_harnesses
    if not harnesses:
        typer.echo("No harnesses enabled: every EVAL__RUN_<harness> flag is false.", err=True)
        raise typer.Exit(code=1)

    # A single harness failing (transient provider error, a bad fixture) should not throw away
    # the spend on the others — run them all, collect failures, and report at the end.
    failed: list[str] = []
    for h in harnesses:
        if clear_cache:
            _clear_cache(h)

        try:
            _run_single_harness(h, tags)
        except Exception as exc:  # report and continue across the suite
            logger.exception("Harness %r failed", h)
            typer.echo(f"Harness '{h}' failed: {exc}", err=True)
            failed.append(h)

    if failed:
        typer.echo(f"\n{len(failed)}/{len(harnesses)} harness(es) failed: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nAll {len(harnesses)} harnesses completed: {', '.join(harnesses)}")


def _run_single_harness(harness: str, tags: list[str] | None) -> None:
    """Bootstrap MLflow and run a single named harness against the labelled corpus."""
    import mlflow

    async def _run() -> None:
        eval_config, seshat_config, run_name = _bootstrap_eval(harness)

        match harness:
            case "grouping":
                from seshat.eval.grouping.entrypoint import run
            case "identification":
                from seshat.eval.identification.entrypoint import run
            case "resolution":
                from seshat.eval.resolution.entrypoint import run
            case "retrieval":
                from seshat.eval.retrieval.entrypoint import run
            case "grounding":
                from seshat.eval.grounding.entrypoint import run
            case _:
                typer.echo(f"Unknown harness '{harness}'. Choose from: {', '.join(_HARNESS_TYPES)}", err=True)
                raise typer.Exit(code=1)

        tag_filter = _parse_tags(tags) if tags is not None else None
        with mlflow.start_run(run_name=run_name):
            await run(eval_config, seshat_config, tag_filter=tag_filter)

    _run_async(_run())


@eval_app.command("clear-cache")
def clear_cache_cmd(
    harness: Annotated[
        str | None,
        typer.Argument(help=f"Harness cache to clear: {' | '.join(_HARNESS_TYPES)}. Omit to clear all."),
    ] = None,
) -> None:
    """Clear cached eval predictions for one harness, or all harnesses when none is given."""
    if harness is None:
        for h in _HARNESS_TYPES:
            _clear_cache(h)
    else:
        _clear_cache(harness)


@eval_app.command("calibrate")
def calibrate_cmd(
    component: Annotated[str, typer.Argument(help=f"Component to calibrate: {' | '.join(_CALIBRATION_TYPES)}")],
    pc_curve: bool = typer.Option(False, "--pc-curve", help="Plot precision-coverage curve (identification only)"),
    p_target: float = typer.Option(0.95, "--p-target", help="Precision target for threshold sweep"),
    ignore_grounding: bool = typer.Option(False, "--ignore-grounding", help="Ignore grounding signal in calibration"),
    clear_cache: Annotated[
        bool,
        typer.Option("--clear-cache", help="Clear this component's prediction cache before calibrating."),
    ] = False,
) -> None:
    """Calibrate eval thresholds and weights for the given component."""
    import mlflow

    if clear_cache:
        _clear_cache(component)

    async def _run() -> None:
        eval_config, seshat_config, run_name = _bootstrap_eval(f"{component}-calibration")

        _kwargs: dict[str, Any] = {"eval_config": eval_config, "seshat_config": seshat_config}
        match component:
            case "retrieval":
                from seshat.eval.calibration.retrieval_entrypoint import run

            case "identification":
                from seshat.eval.calibration.identification_entrypoint import run

                mode = "precision_coverage_curve" if pc_curve else "sweep_threshold"
                _kwargs.update({"mode": mode, "p_target": p_target, "ignore_grounding": ignore_grounding})

            case _:
                typer.echo(f"Unknown component '{component}'. Choose from: {', '.join(_CALIBRATION_TYPES)}", err=True)
                raise typer.Exit(code=1)

        with mlflow.start_run(run_name=run_name):
            await run(**_kwargs)

    _run_async(_run())


@eval_app.command("mlflow")
def mlflow_cmd(
    port: int = typer.Option(5000, "--port", help="Port to serve MLflow UI on"),
) -> None:
    """Start the MLflow tracking server."""
    result = subprocess.run(
        ["uv", "run", "--no-sync", "mlflow", "server", "--port", str(port)],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


@app.command("api")
def api_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (development only)"),
    no_access_log: bool = typer.Option(True, "--no-access-log/--access-log", help="Suppress uvicorn access log"),
) -> None:
    """Start the Seshat API server."""
    import uvicorn

    uvicorn.run(
        "seshat.app.platform.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        access_log=(not no_access_log),
    )


@app.command("worker")
def worker_cmd() -> None:
    """Start the Seshat background worker (standalone mode)."""
    typer.echo("The worker is currently embedded in the API process.", err=True)
    typer.echo("Standalone worker support is not yet implemented.", err=True)
    raise typer.Exit(code=1)


@app.command("migrate")
def migrate_cmd(
    revision: str = typer.Argument(default="head", help="Alembic revision target (default: head)"),
) -> None:
    """Run Alembic database migrations."""
    result = subprocess.run(
        ["uv", "run", "--no-sync", "alembic", "upgrade", revision],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


def _clear_cache(harness: str) -> None:
    """Clear the prediction cache directory for a single harness."""
    from seshat.eval.cache import clear_cache_dir

    if harness not in _HARNESS_TYPES:
        typer.echo(f"Unknown harness '{harness}'. Choose from: {', '.join(_HARNESS_TYPES)}", err=True)
        raise typer.Exit(code=1)

    cache_dir = EvalConfig.cache_dir_for(harness)
    clear_cache_dir(cache_dir)
    typer.echo(f"Cleared eval cache for '{harness}': {cache_dir}")


def _parse_tags(tags: list[str]) -> CorpusTagFilter:
    """Parse ``key=value`` tag strings into a dict, erroring on malformed entries."""
    result: CorpusTagFilter = {}
    for tag in tags:
        if "=" not in tag:
            typer.echo(f"Invalid tag format '{tag}': expected key=value", err=True)
            raise typer.Exit(code=1)
        k, _, v = tag.partition("=")
        result[k] = v
    return result


def _assert_reachable(uri: str, *, label: str, timeout: float = 2.0) -> None:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        typer.echo(f"Cannot reach {label} at {uri} — is the stack up? ({exc})", err=True)
        raise typer.Exit(code=1) from exc


def _ensure_utf8_streams() -> None:
    """Make stdout/stderr tolerate non-ASCII so a stray char cannot crash the CLI.

    MLflow logs a runner emoji at end_run; on a cp1252 Windows console that raised
    UnicodeEncodeError at shutdown. Reconfiguring to utf-8 with backslashreplace degrades
    an unencodable char instead of crashing. No-op on streams lacking reconfigure().
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _bound_mlflow_retries() -> None:
    """Cap MLflow's retry/timeout budgets so a slow tracking server fails fast.

    LangChain autolog is intentionally on during eval (agent traces show in the MLflow UI),
    so a large cold harness queues many trace exports. MLflow has TWO independent paths that
    must both be bounded:
      * sync API calls — MLFLOW_HTTP_REQUEST_MAX_RETRIES (default 7) / _TIMEOUT (default 120)
      * async trace export — MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT (default 500), drained
        at process exit; this is what backed up after a cold 138-call harness and hung ~15min.
    Bounding only the HTTP path is insufficient. setdefault leaves explicit user overrides intact.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
    os.environ.setdefault("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", "20")


def _bootstrap_eval(harness_type: str) -> tuple[EvalConfig, SeshatConfig, str]:
    """Set up MLflow and configs for an eval or calibration run."""
    load_dotenv()
    _patch_httpx_ssl()

    job_id = f"seshat-eval-{harness_type}"
    run_name = f"seshat-eval-{harness_type}-{datetime.now(tz=UTC).isoformat(timespec='minutes')}"

    set_job_id(job_id)
    _ensure_utf8_streams()
    _bound_mlflow_retries()
    eval_config = EvalConfig()
    observability = ObservabilityConfig(mlflow_tracking_uri="http://localhost:5000", mlflow_experiment_name=job_id)

    _assert_reachable(observability.mlflow_tracking_uri, label="MLflow")
    setup_mlflow(observability)

    # Clear any span processor a prior harness registered globally (e.g. identification's
    # node slimmer) so it cannot fire on this harness's differently-shaped prediction spans.
    configure_trace_processors()

    seshat_config = SeshatConfig()
    configure_logging(seshat_config.logging)

    if harness_type == "grounding" and seshat_config.extraction.grounding is None:
        seshat_config = seshat_config._with(extraction=seshat_config.extraction._with(grounding=GroundingLLMConfig()))
        logger.warning("grounding LLM config not found in SeshatConfig, using default grounding config")

    return eval_config, seshat_config, run_name


if __name__ == "__main__":
    app()
