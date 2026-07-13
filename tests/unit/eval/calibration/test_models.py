import pytest

from seshat.eval.calibration.models import RetrievalSweepPoint, RetrievalSweepResult


class TestRetrievalSweepResult:
    def test_points_stored_in_insertion_order(self):
        points = [
            RetrievalSweepPoint(threshold=0.5, recall_at_5=0.6, precision_at_5=0.5, macro_f2=0.6),
            RetrievalSweepPoint(threshold=0.0, recall_at_5=1.0, precision_at_5=0.2, macro_f2=0.5),
            RetrievalSweepPoint(threshold=1.0, recall_at_5=0.0, precision_at_5=0.0, macro_f2=0.0),
        ]
        result = RetrievalSweepResult(points=points, suggested_threshold=0.5)
        assert len(result.points) == 3
        assert result.points[0].threshold == pytest.approx(0.5)
        assert result.points[1].threshold == pytest.approx(0.0)
        assert result.points[2].threshold == pytest.approx(1.0)
