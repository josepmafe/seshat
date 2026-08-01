from __future__ import annotations

import pytest

from seshat.core.config.eval_settings import EvalConfig
from seshat.core.models.enums import EvalHarness


class TestCacheDirFor:
    @pytest.mark.parametrize("harness", list(EvalHarness))
    def test_maps_each_harness_to_its_cache_subdir(self, harness: EvalHarness) -> None:
        result = EvalConfig.cache_dir(harness)
        assert result == EvalConfig._cache_dir / harness.value


class TestCorpusDirFor:
    @pytest.mark.parametrize("harness", list(EvalHarness))
    def test_maps_each_harness_to_its_corpus_subdir(self, harness: EvalHarness) -> None:
        cfg = EvalConfig()
        result = cfg.corpus_dir(harness)
        assert result == cfg.corpus_base_dir / harness.value


class TestEnabledHarnesses:
    def test_all_enabled_by_default(self) -> None:
        cfg = EvalConfig()
        assert cfg.enabled_harnesses == [
            "identification",
            "resolution",
            "vector_search",
            "retrieval",
            "grounding",
            "grouping",
        ]

    def test_disabled_harness_is_excluded(self) -> None:
        cfg = EvalConfig(run_resolution=False, run_grounding=False)
        assert cfg.enabled_harnesses == ["identification", "vector_search", "retrieval", "grouping"]

    def test_none_enabled_returns_empty(self) -> None:
        cfg = EvalConfig(
            run_identification=False,
            run_resolution=False,
            run_retrieval=False,
            run_vector_search=False,
            run_grounding=False,
            run_grouping=False,
        )
        assert cfg.enabled_harnesses == []
