import pytest

from seshat.eval.calibration.models import (
    IdentificationSweepPoint,
    IdentificationSweepResult,
    RetrievalSweepPoint,
    RetrievalSweepResult,
    TypePC,
)
from seshat.models.enums import ConceptType


class TestIdentificationSweepResult:
    def test_round_trips(self):
        point = IdentificationSweepPoint(
            threshold=0.5,
            precision_approved=0.8,
            coverage=0.7,
            per_type={ConceptType.DECISION: TypePC(precision_approved=0.8, coverage=0.7)},
        )
        result = IdentificationSweepResult(points=[point], suggested_threshold=0.5)
        assert result.suggested_threshold == pytest.approx(0.5)
        assert result.points[0].threshold == pytest.approx(0.5)
        assert result.points[0].precision_approved == pytest.approx(0.8)
        assert result.points[0].coverage == pytest.approx(0.7)

    def test_empty_points_allowed(self):
        result = IdentificationSweepResult(points=[], suggested_threshold=0.0)
        assert result.points == []


class TestRetrievalSweepResult:
    def test_round_trips(self):
        point = RetrievalSweepPoint(threshold=0.3, recall_at_5=0.8, precision_at_5=0.4, macro_f2=0.7)
        result = RetrievalSweepResult(points=[point], suggested_threshold=0.3)
        assert result.suggested_threshold == pytest.approx(0.3)
        assert result.points[0].recall_at_5 == pytest.approx(0.8)
        assert result.points[0].precision_at_5 == pytest.approx(0.4)
        assert result.points[0].macro_f2 == pytest.approx(0.7)

    def test_multiple_points_ordered(self):
        points = [
            RetrievalSweepPoint(threshold=0.0, recall_at_5=1.0, precision_at_5=0.2, macro_f2=0.5),
            RetrievalSweepPoint(threshold=0.5, recall_at_5=0.6, precision_at_5=0.5, macro_f2=0.6),
            RetrievalSweepPoint(threshold=1.0, recall_at_5=0.0, precision_at_5=0.0, macro_f2=0.0),
        ]
        result = RetrievalSweepResult(points=points, suggested_threshold=0.0)
        assert len(result.points) == 3
        assert result.points[0].threshold == pytest.approx(0.0)
        assert result.points[2].threshold == pytest.approx(1.0)
