from __future__ import annotations

import asyncio

from seshat.config.settings import EvalConfig, SeshatConfig
from seshat.eval.retrieval.runner import RetrievalEvalRunner
from seshat.observability.mlflow_setup import setup_mlflow
from seshat.utils.log import get_logger
from seshat.vector_store.factory import get_vector_store

logger = get_logger(__name__)


async def run(config: EvalConfig, seshat_config: SeshatConfig, model_id: str | None = None) -> None:
    setup_mlflow(config.observability, disable_autolog=True)

    vector_store = get_vector_store(seshat_config)
    runner = RetrievalEvalRunner(vector_store=vector_store, config=config, model_id=model_id)
    gate = await runner.run()
    logger.info("retrieval eval: passed=%s", gate.passed)


if __name__ == "__main__":
    seshat_config = SeshatConfig()
    eval_config = EvalConfig()
    asyncio.run(run(eval_config, seshat_config))
