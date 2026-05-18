import os

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from seshat.config.settings import ExtractionConfig, IdentificationLLMConfig, ResolutionLLMConfig, VerificationLLMConfig
from seshat.models.enums import LLMProvider
from tests.integration.conftest import _BEDROCK_PROFILE, _azure_available, _bedrock_available

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_OPENAI_MODEL = "gpt-5-nano"
_AZURE_MODEL = "gpt-5-nano"
_BEDROCK_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"


def _make_cheap_llm() -> BaseChatModel:
    if _bedrock_available(profile_name=_BEDROCK_PROFILE):
        return ChatBedrockConverse(model=_BEDROCK_MODEL, temperature=0.0, credentials_profile_name=_BEDROCK_PROFILE)

    if _azure_available():
        return AzureChatOpenAI(azure_deployment=_AZURE_MODEL, api_version="2024-12-01-preview", temperature=0.0)

    if os.environ.get("ANTHROPIC_API_KEY"):
        return ChatAnthropic(model=_ANTHROPIC_MODEL, temperature=0.0)

    return ChatOpenAI(model=_OPENAI_MODEL, temperature=0.0)


def _cheap_llm_config() -> IdentificationLLMConfig:
    if _bedrock_available(profile_name=_BEDROCK_PROFILE):
        return IdentificationLLMConfig(provider=LLMProvider.BEDROCK_CONVERSE, model=_BEDROCK_MODEL)

    if _azure_available():
        return IdentificationLLMConfig(provider=LLMProvider.AZURE_OPENAI, model=_AZURE_MODEL)

    if os.environ.get("ANTHROPIC_API_KEY"):
        return IdentificationLLMConfig(provider=LLMProvider.ANTHROPIC, model=_ANTHROPIC_MODEL)

    return IdentificationLLMConfig(provider=LLMProvider.OPENAI, model=_OPENAI_MODEL)


@pytest.fixture
def cheap_llm() -> BaseChatModel:
    return _make_cheap_llm()


@pytest.fixture
def extraction_config() -> ExtractionConfig:
    return ExtractionConfig(identification=_cheap_llm_config())


@pytest.fixture
def resolution_config() -> ResolutionLLMConfig:
    cfg = _cheap_llm_config()
    return ResolutionLLMConfig(provider=cfg.provider, model=cfg.model)


@pytest.fixture
def verification_config() -> VerificationLLMConfig:
    cfg = _cheap_llm_config()
    return VerificationLLMConfig(provider=cfg.provider, model=cfg.model, max_retries=1)
