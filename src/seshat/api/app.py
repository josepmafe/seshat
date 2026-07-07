from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI
from langchain_core.messages import HumanMessage

from seshat.api.routers import admin, graph, health, identity, jobs
from seshat.api.state import build_app_state
from seshat.app.pipeline.llm_factory import _build_llm
from seshat.core.config.settings import SeshatConfig, get_config
from seshat.core.utils.log import configure_logging, get_logger, set_job_id
from seshat.observability.mlflow_setup import setup_mlflow

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from seshat.core.config.settings import APIConfig, _LLMConfig


logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and return a FastAPI app instance."""
    v1_router = APIRouter(prefix="/v1")
    v1_router.include_router(health.router)
    v1_router.include_router(identity.router)
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

    async with build_app_state(config) as app_state:
        await app_state.job_service.recover_stranded()
        app.state.app_state = app_state
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
    if config.api.skip_llm_ping:
        logger.warning("`skip_llm_ping=True`: LLM ping check bypassed")
        return

    # TODO: add embedding and transcription pings
    faulty_providers: dict[str, list[str]] = {
        "chat": await _ping_chat_models(config),
        "embedding": await _ping_embedding_models(config),
        "transcription": await _ping_transcription_models(config),
    }

    if any(faulty_providers.values()):
        logger.critical("LLM connectivity check failed: %s", json.dumps(faulty_providers, indent=2))
        raise SystemExit(1)


async def _ping_chat_models(config: SeshatConfig) -> list[str]:
    chat_model_configs: list[_LLMConfig | None] = [
        config.extraction.identification,
        config.extraction.identification_self_review.llm,
        config.extraction.grounding,
        config.rag.keyword_extraction_llm,
        config.extraction.resolution,
        config.extraction.resolution_self_review.llm,
    ]

    seen: set[tuple[str, str | None]] = set()
    faulty_providers: list[str] = []
    for llm_cfg in chat_model_configs:
        if llm_cfg is None:
            continue

        key = (llm_cfg.provider, llm_cfg.api_key_secret_key)
        if key in seen:
            continue

        seen.add(key)

        try:
            llm = _build_llm(llm_cfg, config)
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

    return faulty_providers


async def _ping_embedding_models(config: SeshatConfig) -> list[str]:
    return []  # TODO: implement embedding model ping


async def _ping_transcription_models(config: SeshatConfig) -> list[str]:
    return []  # TODO: implement transcription model ping
