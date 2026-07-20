from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.integration.conftest import SKIP_IF_NO_LLM_API
from tests.integration.eval.helpers import make_eval_config, make_identification_meta_scorer

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.agents, pytest.mark.eval, SKIP_IF_NO_LLM_API]


class TestIdentificationMetaScorerIntegration:
    async def test_sweep_end_to_end(self, tmp_path: Path) -> None:
        """Real corpus loader + cheap LLM orchestrator; verifies the full sweep path runs
        without errors and produces one result point per step."""
        config = make_eval_config(tmp_path)
        scorer = make_identification_meta_scorer(config)

        result = await scorer.sweep_threshold()

        corpus_files = list(config.identification_corpus_dir.glob("*.yaml"))
        assert len(result.points) == 11  # step=0.1 → 0.0, 0.1, …, 1.0
        assert len(corpus_files) > 0
        assert result.suggested_threshold is not None
