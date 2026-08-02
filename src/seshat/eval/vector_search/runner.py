from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.app.platform.observability.usage_tracker import track_eval_usage
from seshat.core.models.enums import EvalHarness
from seshat.eval.ranked_search_runner import MinimalConfigSearchEvalRunner

if TYPE_CHECKING:
    from pathlib import Path

    from seshat.app.pipeline.extraction.search_engine import SearchEngine
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import RAGConfig
    from seshat.eval.models import RetrievalCorpusExample
    from seshat.infra.vector_store.base_store import AbstractVectorStore


class VectorSearchEvalRunner(MinimalConfigSearchEvalRunner):
    """Eval runner for the vector_search pass — pure dense-embedding similarity.

    The caller is responsible for passing a minimal RAGConfig (SEMANTIC mode, no keyword
    extraction, no multi-query, no reranker) and a dedicated, empty vector store collection.
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
        rag_config: RAGConfig,
    ) -> None:
        super().__init__(
            harness=EvalHarness.VECTOR_SEARCH,
            score_thresholds=config.vector_search_score_thresholds,
            search_engine=search_engine,
            vector_store=vector_store,
            config=config,
            rag_config=rag_config,
        )

    @track_eval_usage("vector_search")
    async def _run_all_predictions(
        self,
        examples: list[RetrievalCorpusExample],
        threshold: float,
    ) -> tuple[dict[str, list[str]], set[Path], int]:
        return await super()._run_all_predictions(examples, threshold)
