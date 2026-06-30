from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI

from seshat.api.routers import graph, health, jobs
from seshat.api.state import AppState
from seshat.config.settings import SeshatConfig
from seshat.models.enums import JobStatus
from seshat.observability.mlflow_setup import setup_mlflow
from seshat.utils.log import configure_logging, get_logger, set_job_id
from seshat.worker.bootstrap import build_worker_context
from seshat.worker.pipeline_runner import PipelineRunner
from seshat.worker.queue import AsyncioTaskQueue

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from seshat.ops.ledger import OpsLedger


logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and return a FastAPI app instance."""
    v1_router = APIRouter(prefix="/v1")
    v1_router.include_router(health.router)
    v1_router.include_router(jobs.router)
    v1_router.include_router(graph.router)

    app = FastAPI(title="Seshat API", version="0.1.0", lifespan=_lifespan)
    app.include_router(v1_router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    set_job_id("api")

    config = SeshatConfig()
    configure_logging(config.logging)

    _emit_config_warnings(config)
    _check_eval_gate(config)

    setup_mlflow(config.observability)

    async with build_worker_context(config) as ctx:
        await _check_stranded_jobs(ctx.ops)

        runner = PipelineRunner.from_context(ctx)
        queue = AsyncioTaskQueue()
        app.state.app_state = AppState.from_context(ctx, config, runner, queue)

        yield


def _emit_config_warnings(config: SeshatConfig) -> None:
    if config.extraction.grounding is None:
        logger.warning("`grounding=None`: heuristics-only confidence scoring for identified nodes.")


def _check_eval_gate(config: SeshatConfig) -> None:
    if config.skip_eval_gate:
        logger.warning("`skip_eval_gate=True`: eval gate check bypassed")
        return

    gate_path = config.eval_gate_path
    if not gate_path.exists():
        logger.critical("%s not found. Run 'seshat eval' first.", gate_path)
        raise SystemExit(1)

    gate = json.loads(gate_path.read_text())
    if not gate.get("passed"):
        logger.critical("eval gate not passed. Run 'seshat eval' first.")
        raise SystemExit(1)


async def _check_stranded_jobs(ops: OpsLedger) -> None:
    stranded = await ops.get_stranded_writing_jobs()
    for job_id in stranded:
        await ops.fail_job(job_id, JobStatus.WRITING, "Server crash during write", recoverable=True)
        logger.warning("Startup recovery: marked stranded job %s as FAILED", job_id)


if __name__ == "__main__":
    import asyncio

    import uvicorn

    from seshat.utils.log import configure_logging

    async def _serve() -> None:
        config = uvicorn.Config(create_app(), host="0.0.0.0", port=8000, log_config=None)
        await uvicorn.Server(config).serve()

    asyncio.run(_serve())
