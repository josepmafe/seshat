from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.blob_store.factory import get_blob_store
from seshat.eval.identification.runner import IdentificationEvalRunner
from seshat.eval.mlflow_logging import log_eval_model
from seshat.knowledge_store.factory import get_kb_store
from seshat.pipeline.bootstrap import build_orchestrator
from seshat.utils.log import get_logger
from seshat.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.config.eval_settings import EvalConfig
    from seshat.config.settings import SeshatConfig
    from seshat.eval.corpus_tags import CorpusTagFilter

logger = get_logger(__name__)


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig, tag_filter: CorpusTagFilter | None = None):
    kb_store = get_kb_store(seshat_config)
    await kb_store.connect()

    vector_store = get_vector_store(seshat_config)

    blob_store = get_blob_store(seshat_config)
    await blob_store.connect()

    llm_cfg = seshat_config.extraction.identification
    logger.info("LLM provider=%r model=%r temperature=%s", llm_cfg.provider.value, llm_cfg.model, llm_cfg.temperature)

    try:
        orchestrator = build_orchestrator(seshat_config, kb_store, vector_store, blob_store)
        model_id = log_eval_model(
            "seshat-identification-agent", inference_component=orchestrator._identification_registry, llm_config=llm_cfg
        )

        runner = IdentificationEvalRunner(orchestrator=orchestrator, config=eval_config)
        gate = await runner.run(tag_filter=tag_filter, model_id=model_id)

        logger.info("identification eval: passed=%s", gate.passed)

    finally:
        await kb_store.close()
        await blob_store.close()
