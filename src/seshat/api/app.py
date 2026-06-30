from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI
from langchain_core.messages import HumanMessage

from seshat.api.routers import admin, graph, health, jobs
from seshat.api.state import AppState
from seshat.config.settings import SeshatConfig, get_config
from seshat.models.enums import JobStatus
from seshat.observability.mlflow_setup import setup_mlflow
from seshat.pipeline.llm_factory import _build_llm
from seshat.utils.log import configure_logging, get_logger, set_job_id
from seshat.worker.bootstrap import build_worker_context
from seshat.worker.pipeline_runner import PipelineRunner
from seshat.worker.queue import AsyncioTaskQueue

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from seshat.config.settings import APIConfig, _LLMConfig
    from seshat.ops.ledger import OpsLedger


logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and return a FastAPI app instance."""
    v1_router = APIRouter(prefix="/v1")
    v1_router.include_router(health.router)
    v1_router.include_router(jobs.router)
    v1_router.include_router(graph.router)
    v1_router.include_router(admin.router)

    app = FastAPI(title="Seshat API", version="0.1.0", lifespan=_lifespan)
    app.include_router(v1_router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    set_job_id("api")

    config = get_config()
    configure_logging(config.logging)

    _emit_config_warnings(config)
    _check_eval_gate(config.api)

    setup_mlflow(config.observability)
    await _ping_llms(config)

    async with build_worker_context(config) as ctx:
        await _check_stranded_jobs(ctx.ops)

        runner = PipelineRunner.from_context(ctx)
        queue = AsyncioTaskQueue()
        app.state.app_state = AppState.from_context(ctx, config, runner, queue)

        yield


def _emit_config_warnings(config: SeshatConfig) -> None:
    if config.extraction.grounding is None:
        logger.warning("`grounding=None`: heuristics-only confidence scoring for identified nodes.")


def _check_eval_gate(config: APIConfig) -> None:
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


async def _ping_llms(config: SeshatConfig) -> None:
    """Verify connectivity to all configured LLM providers. Raises SystemExit(1) on failure."""
    llm_configs: list[_LLMConfig | None] = [
        config.extraction.identification,
        config.extraction.identification_self_review.llm,
        config.extraction.grounding,
        config.rag.keyword_extraction_llm,
        config.extraction.resolution,
        config.extraction.resolution_self_review.llm,
    ]

    seen: set[tuple[str, str | None]] = set()
    faulty_providers: list[str] = []
    for llm_cfg in llm_configs:
        if llm_cfg is None:
            continue

        key = (llm_cfg.provider, llm_cfg.api_key_secret_key)
        if key in seen:
            continue

        seen.add(key)
        llm = _build_llm(llm_cfg, config)
        try:
            await llm.ainvoke([HumanMessage(content="ping")], max_tokens=1)
            logger.debug("LLM reachable: provider=%s model=%s", llm_cfg.provider, llm_cfg.model)
        except Exception as exc:
            logger.warning(
                "LLM provider unreachable at startup: provider=%s model=%s — %s: %s",
                llm_cfg.provider,
                llm_cfg.model,
                type(exc).__name__,
                exc,
            )
            faulty_providers.append(llm_cfg.provider)

    if faulty_providers:
        logger.critical("LLM connectivity check failed for providers: %s", ", ".join(faulty_providers))
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
