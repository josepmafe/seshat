import math

from seshat.config.settings import ConfidenceWeights
from seshat.models.nodes import ConfidenceBreakdown
from seshat.pipeline.extraction.weighted_scorer import compute_confidence


class TestComputeConfidence:
    def test_result_in_zero_one_range(self):
        weights = ConfidenceWeights()
        bd = compute_confidence(verification=0.8, heuristics=1.0, weights=weights)
        assert isinstance(bd, ConfidenceBreakdown)
        assert 0.0 <= bd.final <= 1.0

    def test_all_signals_active(self):
        weights = ConfidenceWeights(verification=0.70, heuristics=0.30)
        bd = compute_confidence(verification=1.0, heuristics=0.6, weights=weights)
        expected = (0.70 * 1.0 + 0.30 * 0.6) / 1.0
        assert math.isclose(bd.final, expected)
        assert bd.verification == 1.0
        assert bd.heuristics == 0.6

    def test_verification_unavailable_redistributes(self):
        weights = ConfidenceWeights(verification=0.70, heuristics=0.30)
        bd = compute_confidence(verification=None, heuristics=0.6, weights=weights)
        assert math.isclose(bd.final, 0.6)
        assert bd.verification is None
