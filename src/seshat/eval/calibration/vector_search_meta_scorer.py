from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.app.platform.observability.usage_tracker import track_eval_usage
from seshat.core.models.enums import EvalHarness
from seshat.eval.calibration.ranked_search_meta_scorer import MinimalConfigSearchMetaScorer

if TYPE_CHECKING:
    from seshat.app.pipeline.extraction.search_engine import SearchEngine
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import RAGConfig
    from seshat.eval.calibration.ranked_search_meta_scorer import _Cache
    from seshat.infra.vector_store.base_store import AbstractVectorStore


class VectorSearchMetaScorer(MinimalConfigSearchMetaScorer):
    """Sweeps the dense-leg cosine-similarity threshold for the vector_search harness.

    The caller is responsible for passing a minimal RAGConfig (SEMANTIC mode, no keyword
    extraction, no multi-query, no reranker) — same requirement as VectorSearchEvalRunner.
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
        rag_config: RAGConfig,
        step: float = 0.005,
    ) -> None:
        from seshat.eval.vector_search.runner import VectorSearchEvalRunner

        super().__init__(
            harness=EvalHarness.VECTOR_SEARCH,
            runner_factory=lambda: VectorSearchEvalRunner(
                search_engine=search_engine,
                vector_store=vector_store,
                config=config,
                rag_config=rag_config,
            ),
            config=config,
            search_mode_hash=search_engine.fingerprint(),
            step=step,
        )

    @track_eval_usage("vector_search")
    async def _build_cache(self) -> _Cache:
        return await super()._build_cache()
