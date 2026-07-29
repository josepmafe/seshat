import pytest
from pydantic import ValidationError

from seshat.core.config.settings import (
    APIConfig,
    ExtractionConfig,
    GroundingLLMConfig,
    IdentificationLLMConfig,
    ReflectiveLLMConfig,
    SecretsConfig,
    SeshatConfig,
    SeshatConfigOverride,
    TranscriptionConfig,
    get_request_settings,
)
from seshat.core.models.enums import LLMProvider, SecretsProvider


class TestReflectiveLLMConfig:
    def test_enabled_with_defaults(self):
        cfg = ReflectiveLLMConfig(enabled=True)
        assert cfg.enabled is True

    def test_llm_override_accepted(self):
        llm = IdentificationLLMConfig()
        cfg = ReflectiveLLMConfig(enabled=True, llm=llm)
        assert cfg.llm is llm


class TestGroundingModelValidator:
    def test_same_provider_raises(self):
        with pytest.raises(ValidationError, match=r"grounding.provider"):
            ExtractionConfig(
                identification=IdentificationLLMConfig(provider=LLMProvider.ANTHROPIC),
                grounding=GroundingLLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model="claude-haiku-4-5-20251001",
                ),
            )


class TestGetRequestSettings:
    def test_none_override_returns_singleton(self, monkeypatch, minimal_config: SeshatConfig):
        monkeypatch.setattr("seshat.core.config.settings._config", minimal_config)
        assert get_request_settings(None) is minimal_config

    def test_override_applies_and_preserves_unset_fields(self, monkeypatch, minimal_config: SeshatConfig):
        monkeypatch.setattr("seshat.core.config.settings._config", minimal_config)
        result = get_request_settings(SeshatConfigOverride(extraction=ExtractionConfig(confidence_threshold=0.5)))
        assert result.extraction.confidence_threshold == 0.5
        assert result.extraction.identification.provider == minimal_config.extraction.identification.provider
        assert result.extraction.identification.model == minimal_config.extraction.identification.model

    def test_partial_override_preserves_non_default_base_values(self, monkeypatch):
        """Overriding one extraction field must not revert sibling fields to their
        defaults when the base config carries non-default values for those fields."""
        monkeypatch.setenv("postgres_url", "postgresql://seshat:seshat@localhost:5432/seshat")
        # Base has a non-default max_output_tokens on the identification LLM config.
        base = SeshatConfig(
            _env_file=None,  # type: ignore[call-arg]
            secrets=SecretsConfig(provider=SecretsProvider.ENV),
            transcription=TranscriptionConfig(max_retries=1),
            extraction=ExtractionConfig(identification=IdentificationLLMConfig(max_output_tokens=1024)),
            api=APIConfig(max_jobs_per_user_per_hour=5),
        )
        monkeypatch.setattr("seshat.core.config.settings._config", base)

        # Override only confidence_threshold, the other fields should survive unchanged:
        # - transcription.max_retries (another config class)
        # - extraction.identification.max_output_tokens (same config class)
        # - api.max_jobs_per_user_per_hour (nested api config field)
        result = get_request_settings(SeshatConfigOverride(extraction=ExtractionConfig(confidence_threshold=0.5)))

        # should be overridden
        assert result.extraction.confidence_threshold == 0.5
        # must not revert to default
        assert result.transcription.max_retries == 1
        assert result.extraction.identification.max_output_tokens == 1024
        assert result.api.max_jobs_per_user_per_hour == 5

    def test_explicit_null_section_treated_as_unset(self, monkeypatch, minimal_config: SeshatConfig):
        """A section present in the payload with value null (as opposed to omitted) must not crash."""
        monkeypatch.setattr("seshat.core.config.settings._config", minimal_config)
        overrides = SeshatConfigOverride.model_validate({"extraction": None})
        assert overrides.model_fields_set == {"extraction"}

        assert get_request_settings(overrides) == minimal_config

    def test_nested_submodel_override_preserves_type_and_sibling_fields(self, monkeypatch):
        """Overriding a nested sub-model field (e.g. extraction.identification) must merge it
        field-by-field rather than replacing it with a plain dict, so its type and untouched
        sibling fields (both on the sub-model and on the section that holds it) survive."""
        monkeypatch.setenv("postgres_url", "postgresql://seshat:seshat@localhost:5432/seshat")
        base = SeshatConfig(
            _env_file=None,  # type: ignore[call-arg]
            secrets=SecretsConfig(provider=SecretsProvider.ENV),
            extraction=ExtractionConfig(identification=IdentificationLLMConfig(max_output_tokens=1024)),
        )
        monkeypatch.setattr("seshat.core.config.settings._config", base)

        overrides = SeshatConfigOverride(
            extraction=ExtractionConfig(identification=IdentificationLLMConfig(temperature=0.9))
        )
        result = get_request_settings(overrides)

        assert isinstance(result.extraction.identification, IdentificationLLMConfig)
        assert result.extraction.identification.temperature == 0.9
        # sibling field on the same sub-model must survive, not revert to its default
        assert result.extraction.identification.max_output_tokens == 1024
        # sibling field on the section holding the sub-model must survive unchanged
        assert result.extraction.confidence_threshold == base.extraction.confidence_threshold
