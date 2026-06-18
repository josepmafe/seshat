from __future__ import annotations

import json
from datetime import date

import yaml
from pydantic import BaseModel, Field, ValidationError

_ACCEPTED_EXTENSIONS = frozenset({"yaml", "yml", "json"})


class TextValidationError(Exception):
    pass


class ParsedTextInput(BaseModel):
    meeting_date: date = Field(alias="date")
    content: str
    participants: list[str] | None = None

    model_config = {"populate_by_name": True}


class TextValidator:
    @classmethod
    def parse(cls, raw: bytes, filename: str) -> ParsedTextInput:
        ext = filename.rsplit(".", 1)[-1].lower()

        match ext:
            case "yaml" | "yml":
                data = cls._parse_yaml(raw)
            case "json":
                data = cls._parse_json(raw)
            case _:
                raise TextValidationError(
                    f"Unsupported file extension {ext!r}. Accepted: {', '.join(sorted(_ACCEPTED_EXTENSIONS))}"
                )

        if not isinstance(data, dict):
            raise TextValidationError(
                f"'{filename}' must contain a key-value mapping at the top level, got {type(data).__name__}"
            )

        try:
            return ParsedTextInput.model_validate(data)
        except ValidationError as exc:
            raise TextValidationError(str(exc)) from exc

    @staticmethod
    def _parse_yaml(raw: bytes) -> object:
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise TextValidationError(f"Invalid YAML: {exc}") from exc

    @staticmethod
    def _parse_json(raw: bytes) -> object:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TextValidationError(f"Invalid JSON: {exc}") from exc
