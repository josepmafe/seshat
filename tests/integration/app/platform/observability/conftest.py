import pytest
from langchain_openai import AzureOpenAIEmbeddings

from seshat.app.platform.observability.usage_tracker import TrackingEmbeddings
from seshat.core.config.settings import IdentificationLLMConfig
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API
from tests.integration.helpers import cheap_identification_config


@pytest.fixture(scope="module")
def identification_config() -> IdentificationLLMConfig:
    return cheap_identification_config()


@pytest.fixture(scope="module")
@SKIP_IF_NO_EMBEDDINGS_API
def azure_embeddings() -> TrackingEmbeddings:
    return TrackingEmbeddings(AzureOpenAIEmbeddings())
