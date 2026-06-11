from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

from seshat.eval.calibration.retrieval_meta_scorer import RetrievalMetaScorer
from seshat.eval.mlflow_logging import log_retrieval_model
from seshat.utils.log import get_logger
from seshat.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.config.eval_settings import EvalConfig
    from seshat.config.settings import SeshatConfig

logger = get_logger(__name__)


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig) -> None:
    log_retrieval_model("seshat-retrieval", seshat_config.vector_index)

    vector_store = get_vector_store(seshat_config)
    scorer = RetrievalMetaScorer(vector_store=vector_store, config=eval_config)

    logger.info("Sweeping thresholds...")
    result = await scorer.sweep_threshold()

    suggested = result.suggested_threshold
    logger.info("Suggested threshold: %.2f", suggested)
    logger.info("Eval harness: set EVAL__RETRIEVAL_SCORE_THRESHOLD=%.2f in .env", suggested)
    logger.info("Production pipeline: set RAG__MIN_SIMILARITY_SCORE=%.2f in .env", suggested)

    metrics = next(p for p in result.points if p.threshold == suggested)
    run = mlflow.active_run()
    run_id = run.info.run_id if run else None
    mlflow.log_metrics(metrics.model_dump(), run_id=run_id)
