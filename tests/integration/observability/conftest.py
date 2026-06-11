import pytest
from langchain_core.language_models import BaseChatModel

from seshat.config.settings import IdentificationLLMConfig
from tests.integration.helpers import cheap_identification_config, make_cheap_llm


@pytest.fixture
def cheap_llm() -> BaseChatModel:
    return make_cheap_llm()


@pytest.fixture
def identification_config() -> IdentificationLLMConfig:
    return cheap_identification_config()
