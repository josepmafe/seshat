from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

from seshat.app.pipeline.bootstrap import get_search_engine
from seshat.core.models.enums import SearchMode
from seshat.core.utils.log import get_logger
from seshat.eval.calibration.sparse_search_meta_scorer import SparseSearchMetaScorer
from seshat.eval.mlflow_logging import log_retrieval_model
from seshat.eval.sparse_search.entrypoint import _minimal_rag_config
from seshat.infra.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import SeshatConfig

logger = get_logger(__name__)


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig) -> None:
    """Sweep the sparse-leg ts_rank_cd threshold for sparse_search, isolated from any LLM component."""
    minimal_rag_config = _minimal_rag_config(seshat_config.rag)
    minimal_config = seshat_config._with(rag=minimal_rag_config)

    log_retrieval_model("seshat-sparse-search", minimal_config.vector_index)

    vector_store = get_vector_store(minimal_config)
    search_engine = get_search_engine(minimal_config, vector_store)
    scorer = SparseSearchMetaScorer(
        search_engine=search_engine,
        vector_store=vector_store,
        config=eval_config,
        rag_config=minimal_rag_config,
    )

    logger.info("Sweeping thresholds for sparse_search (KEYWORD mode)...")
    result = await scorer.sweep_threshold()

    suggested = result.suggested_threshold
    logger.info("Suggested threshold: %.2f", suggested)
    logger.info("Set EVAL__SPARSE_SEARCH_SCORE_THRESHOLDS__KEYWORD=%.2f in .env", suggested)

    metrics = next(p for p in result.points if p.threshold == suggested)
    mlflow.log_metrics(metrics.model_dump())
    mlflow.log_param("sparse_search.search_mode", SearchMode.KEYWORD.value)
