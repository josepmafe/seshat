from __future__ import annotations

from seshat.app.pipeline.extraction.reranker import AbstractReranker, CohereReranker, VoyageReranker, reranker_factory
from seshat.core.config.settings import RerankerConfig
from seshat.core.models.enums import RerankerProvider
from tests.helpers import make_node


class _FakeReranker(AbstractReranker):
    async def _rerank(self, query, nodes):
        return nodes


def _nodes(*seeds: str):
    return [make_node(s) for s in seeds]


class TestAbstractRerankerTopN:
    async def test_top_n_none_returns_all(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="x", top_n=None)
        out = await _FakeReranker(cfg, "key").rerank("q", _nodes("a", "b", "c"))
        assert len(out) == 3

    async def test_top_n_truncates(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="x", top_n=2)
        out = await _FakeReranker(cfg, "key").rerank("q", _nodes("a", "b", "c"))
        assert len(out) == 2


class TestBuildReranker:
    def test_cohere_returns_cohere_reranker(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5")
        assert isinstance(reranker_factory(cfg, "key"), CohereReranker)

    def test_voyage_returns_voyage_reranker(self):
        cfg = RerankerConfig(provider=RerankerProvider.VOYAGE, model="rerank-2")
        assert isinstance(reranker_factory(cfg, "key"), VoyageReranker)
