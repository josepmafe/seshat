import json
from datetime import date

import pytest
import yaml

from seshat.app.pipeline.ingestion.text_validator import ParsedTextInput, TextValidationError, TextValidator


class TestTextValidator:
    def _yaml_bytes(self, data: dict) -> bytes:
        return yaml.dump(data).encode()

    def _json_bytes(self, data: dict) -> bytes:
        return json.dumps(data).encode()

    def test_valid_yaml_with_participants(self):
        raw = self._yaml_bytes(
            {
                "date": "2026-04-21",
                "content": "Alice: We decided to use PostgreSQL.",
                "participants": ["Alice", "Bob"],
            }
        )
        result = TextValidator.parse(raw, filename="meeting.yaml")
        assert isinstance(result, ParsedTextInput)
        assert result.content == "Alice: We decided to use PostgreSQL."
        assert result.participants == ["Alice", "Bob"]
        assert result.meeting_date == date(2026, 4, 21)

    def test_valid_json_without_participants(self):
        raw = self._json_bytes(
            {
                "date": "2026-04-21",
                "content": "We decided to use PostgreSQL.",
            }
        )
        result = TextValidator.parse(raw, filename="meeting.json")
        assert result.participants is None

    def test_missing_date_raises(self):
        raw = self._yaml_bytes({"content": "hello"})
        with pytest.raises(TextValidationError, match="date"):
            TextValidator.parse(raw, filename="meeting.yaml")

    def test_missing_content_raises(self):
        raw = self._yaml_bytes({"date": "2026-04-21"})
        with pytest.raises(TextValidationError, match="content"):
            TextValidator.parse(raw, filename="meeting.yaml")

    def test_invalid_yaml_raises(self):
        with pytest.raises(TextValidationError):
            TextValidator.parse(b"[invalid: yaml: {{", filename="meeting.yaml")

    def test_unsupported_extension_raises(self):
        with pytest.raises(TextValidationError, match="Unsupported"):
            TextValidator.parse(b"{}", filename="meeting.txt")
