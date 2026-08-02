from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.app.pipeline.bootstrap import get_search_engine
from seshat.core.models.enums import SearchMode
from seshat.core.utils.log import get_logger
from seshat.eval.mlflow_logging import log_retrieval_model
from seshat.eval.vector_search.runner import VectorSearchEvalRunner
from seshat.infra.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import RAGConfig, SeshatConfig, VectorIndexConfig
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.infra.vector_store.base_store import AbstractVectorStore


logger = get_logger(__name__)

# Dedicated collection for vector_search eval — isolated from production nodes so seed/teardown
# per corpus example cannot corrupt or be corrupted by the live vector store.
_EVAL_COLLECTION = "seshat-vector-search-eval"


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig, tag_filter: CorpusTagFilter | None = None) -> None:
    """Run the vector_search eval pass: pure dense-embedding similarity, isolated from any LLM component.

    Forces a minimal RAGConfig (SEMANTIC mode, no keyword extraction, no multi-query, no reranker) so
    SearchEngine dispatches straight to vector_store.search_dense() with zero LLM calls — this harness
    measures embedding quality alone, decoupled from the keyword_extraction/multi_query/reranker harnesses.
    """
    minimal_rag_config = _minimal_rag_config(seshat_config.rag)
    minimal_config = seshat_config._with(rag=minimal_rag_config)

    vector_store, index_config = _ensure_clean_vector_store(minimal_config)
    model_id = log_retrieval_model("seshat-vector-search", index_config)

    logger.info("vector_search eval: model_id=%s", model_id)

    search_engine = get_search_engine(minimal_config, vector_store)
    runner = VectorSearchEvalRunner(
        search_engine=search_engine,
        vector_store=vector_store,
        config=eval_config,
        rag_config=minimal_rag_config,
    )
    gate = await runner.run(tag_filter=tag_filter, model_id=model_id)

    logger.info("vector_search eval: passed=%s", gate.passed)


def _minimal_rag_config(rag_config: RAGConfig) -> RAGConfig:
    """Force SEMANTIC mode with every LLM-backed retrieval component disabled."""
    return rag_config._with(
        search_mode=SearchMode.SEMANTIC,
        multi_query=None,
    )


def _ensure_clean_vector_store(seshat_config: SeshatConfig) -> tuple[AbstractVectorStore, VectorIndexConfig]:
    """Ensure the vector store is clean before starting eval, to prevent test contamination from previous runs."""
    vector_index_cfg = seshat_config.vector_index._with(collection=_EVAL_COLLECTION)
    eval_config = seshat_config._with(vector_index=vector_index_cfg)
    return get_vector_store(eval_config), vector_index_cfg
