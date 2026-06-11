from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.eval.mlflow_logging import log_retrieval_model
from seshat.eval.retrieval.runner import RetrievalEvalRunner
from seshat.utils.log import get_logger
from seshat.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.config.eval_settings import EvalConfig
    from seshat.config.settings import SeshatConfig
    from seshat.eval.corpus_tags import CorpusTagFilter

logger = get_logger(__name__)


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig, tag_filter: CorpusTagFilter | None = None) -> None:
    vector_store = get_vector_store(seshat_config)
    model_id = log_retrieval_model("seshat-retrieval", seshat_config.vector_index)

    runner = RetrievalEvalRunner(vector_store=vector_store, config=eval_config)
    gate = await runner.run(tag_filter=tag_filter, model_id=model_id)

    logger.info("retrieval eval: passed=%s", gate.passed)
