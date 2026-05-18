import pytest
from pydantic import ValidationError

from seshat.models.nodes import ConfidenceBreakdown
from tests.helpers import make_node as _make_node


class TestConfidenceBreakdown:
    def test_null_heuristics_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(heuristics=None, final=0.5)  # type: ignore[arg-type]


class TestConfidenceRange:
    def test_node_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            _make_node(confidence=1.1)

    def test_node_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            _make_node(confidence=-0.1)

    def test_breakdown_final_above_1_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(heuristics=0.8, final=1.5)

    def test_breakdown_heuristics_below_0_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(heuristics=-0.1, final=0.5)
