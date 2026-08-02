from __future__ import annotations

from seshat.core.config.settings import MultiQueryConfig, RAGConfig, RerankerConfig
from seshat.core.models.enums import LLMProvider, RerankerProvider, SearchMode
from seshat.eval.sparse_search.entrypoint import _minimal_rag_config


class TestMinimalRagConfig:
    def test_forces_keyword_mode(self):
        rag_config = RAGConfig(search_mode=SearchMode.HYBRID)
        result = _minimal_rag_config(rag_config)
        assert result.search_mode == SearchMode.KEYWORD

    def test_disables_keyword_extraction(self):
        from seshat.core.config.settings import _LLMConfig

        rag_config = RAGConfig(keyword_extraction=_LLMConfig(provider=LLMProvider.OPENAI, model="gpt-5.4-nano"))
        result = _minimal_rag_config(rag_config)
        assert result.keyword_extraction is None

    def test_disables_multi_query_llm(self):
        rag_config = RAGConfig(multi_query=MultiQueryConfig(provider=LLMProvider.OPENAI, model="gpt-5.4-nano"))
        result = _minimal_rag_config(rag_config)
        assert result.multi_query is None

    def test_disables_reranker(self):
        rag_config = RAGConfig(reranker=RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5"))
        result = _minimal_rag_config(rag_config)
        assert result.reranker is None

    def test_preserves_top_k(self):
        rag_config = RAGConfig(top_k=7)
        result = _minimal_rag_config(rag_config)
        assert result.top_k == 7
