import pytest
from pydantic import ValidationError

from seshat.models.nodes import ConfidenceBreakdown
from seshat.models.transcript import Turn
from tests.helpers import make_node as _make_node


class TestConfidenceBreakdown:
    def test_null_heuristics_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(heuristics=None, final=0.5)  # type: ignore[arg-type]


class TestTurnValidation:
    def test_negative_start_seconds_raises(self):
        with pytest.raises(ValidationError):
            Turn(text="hello", start_seconds=-1.0)

    def test_negative_end_seconds_raises(self):
        with pytest.raises(ValidationError):
            Turn(text="hello", end_seconds=-0.1)

    def test_none_offsets_accepted(self):
        t = Turn(text="hello")
        assert t.start_seconds is None
        assert t.end_seconds is None


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
