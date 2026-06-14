from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.eval.mlflow_logging import log_retrieval_model
from seshat.eval.retrieval.runner import RetrievalEvalRunner
from seshat.utils.log import get_logger
from seshat.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.config.eval_settings import EvalConfig
    from seshat.config.settings import SeshatConfig, VectorIndexConfig
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.vector_store.base_store import AbstractVectorStore


logger = get_logger(__name__)

# Dedicated collection for retrieval eval — isolated from production nodes so seed/teardown
# per corpus example cannot corrupt or be corrupted by the live vector store.
_EVAL_COLLECTION = "seshat-retrieval-eval"


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig, tag_filter: CorpusTagFilter | None = None) -> None:
    vector_store, index_config = _ensure_clean_vector_store(seshat_config)
    model_id = log_retrieval_model("seshat-retrieval", index_config)

    runner = RetrievalEvalRunner(vector_store=vector_store, config=eval_config)
    gate = await runner.run(tag_filter=tag_filter, model_id=model_id)

    logger.info("retrieval eval: passed=%s", gate.passed)


def _ensure_clean_vector_store(seshat_config: SeshatConfig) -> tuple[AbstractVectorStore, VectorIndexConfig]:
    """Ensure the vector store is clean before starting eval, to prevent test contamination from previous runs."""
    vector_index_cfg = seshat_config.vector_index.model_copy(update={"collection": _EVAL_COLLECTION})
    vector_store = get_vector_store(seshat_config.model_copy(update={"vector_index": vector_index_cfg}))
    return vector_store, vector_index_cfg
