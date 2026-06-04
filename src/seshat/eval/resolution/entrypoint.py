from __future__ import annotations

import asyncio

from seshat.blob_store.factory import get_blob_store
from seshat.config.eval_settings import EvalConfig
from seshat.config.settings import SeshatConfig
from seshat.eval.resolution.runner import ResolutionEvalRunner
from seshat.knowledge_store.factory import get_kb_store
from seshat.observability.mlflow_setup import setup_mlflow
from seshat.pipeline.bootstrap import build_orchestrator
from seshat.utils.log import get_logger
from seshat.vector_store.factory import get_vector_store

logger = get_logger(__name__)


async def run(config: EvalConfig, seshat_config: SeshatConfig, model_id: str | None = None) -> None:
    setup_mlflow(config.observability, disable_autolog=True)

    kb_store = get_kb_store(seshat_config)
    await kb_store.connect()
    vector_store = get_vector_store(seshat_config)
    blob_store = get_blob_store(seshat_config)
    await blob_store.connect()

    try:
        orchestrator = build_orchestrator(seshat_config, kb_store, vector_store, blob_store)
        runner = ResolutionEvalRunner(orchestrator=orchestrator, config=config, model_id=model_id)
        gate = await runner.run()
        logger.info("resolution eval: passed=%s", gate.passed)
    finally:
        await kb_store.close()
        await blob_store.close()


if __name__ == "__main__":
    seshat_config = SeshatConfig()
    eval_config = EvalConfig()
    asyncio.run(run(eval_config, seshat_config))
