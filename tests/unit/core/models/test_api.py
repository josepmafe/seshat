from datetime import date, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from seshat.core.models.api_graph import ManualNodeCreate, NodeFilter, SearchResult
from seshat.core.models.api_jobs import ApproveRequest, BulkApproveRule
from seshat.core.models.enums import ConceptType


class TestNodeFilterValidation:
    def test_min_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            NodeFilter(min_confidence=1.1)

    def test_min_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            NodeFilter(min_confidence=-0.1)

    def test_meeting_date_to_in_future_raises(self):
        with pytest.raises(ValidationError):
            NodeFilter(meeting_date_to=date.today() + timedelta(days=1))

    def test_meeting_date_from_after_to_raises(self):
        with pytest.raises(ValidationError):
            NodeFilter(
                meeting_date_from=date(2026, 5, 1),
                meeting_date_to=date(2026, 4, 1),
            )


class TestSearchResultValidation:
    def test_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(node_id=UUID("00000000-0000-0000-0000-000000000001"), score=-0.1)

    def test_non_uuid_node_id_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(node_id="not-a-uuid", score=0.5)


class TestBulkApproveRuleValidation:
    def test_threshold_below_05_raises(self):
        with pytest.raises(ValidationError):
            BulkApproveRule(threshold=0.4)

    def test_threshold_equal_1_raises(self):
        with pytest.raises(ValidationError):
            BulkApproveRule(threshold=1.0)


class TestApproveRequestValidation:
    def test_empty_payload_raises(self):
        with pytest.raises(ValidationError, match="ApproveRequest"):
            ApproveRequest()


class TestManualNodeCreateValidation:
    def _base(self, **kwargs) -> dict:
        return {"type": ConceptType.DECISION, "title": "T", "description": "D", **kwargs}

    def test_source_quote_without_blob_key_raises(self):
        with pytest.raises(ValidationError, match="co-required"):
            ManualNodeCreate(**self._base(source_quote="some quote"))

    def test_blob_key_without_source_quote_raises(self):
        with pytest.raises(ValidationError, match="co-required"):
            ManualNodeCreate(**self._base(blob_key="blobs/key.txt"))

    def test_both_fields_together_is_valid(self):
        node = ManualNodeCreate(**self._base(source_quote="quote", blob_key="blobs/key.txt"))
        assert node.source_quote == "quote"
        assert node.blob_key == "blobs/key.txt"

    def test_neither_field_is_valid(self):
        node = ManualNodeCreate(**self._base())
        assert node.source_quote is None
        assert node.blob_key is None


class TestSearchResultValid:
    def test_valid_search_result(self):
        result = SearchResult(node_id=UUID("00000000-0000-0000-0000-000000000001"), score=0.95)
        assert result.score == 0.95

    def test_score_zero_allowed(self):
        result = SearchResult(node_id=UUID("00000000-0000-0000-0000-000000000001"), score=0.0)
        assert result.score == 0.0
