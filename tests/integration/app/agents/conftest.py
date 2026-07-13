import pytest

from seshat.core.config.settings import ExtractionConfig, GroundingLLMConfig, ResolutionLLMConfig
from tests.integration.helpers import (
    cheap_grounding_config,
    cheap_identification_config,
    cheap_resolution_config,
)


@pytest.fixture(scope="module")
def extraction_config() -> ExtractionConfig:
    return ExtractionConfig(identification=cheap_identification_config())


@pytest.fixture(scope="module")
def resolution_config() -> ResolutionLLMConfig:
    return cheap_resolution_config()


@pytest.fixture(scope="module")
def grounding_config() -> GroundingLLMConfig:
    return cheap_grounding_config()
