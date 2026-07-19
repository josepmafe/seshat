from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from seshat.app.agents.base import RetryExhaustedError, _BaseAgent
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.core.config.settings import _LLMConfig

logger = get_logger(__name__)

_MULTI_QUERY_PROMPT_TEMPLATE = """\
You are generating alternative search queries for a knowledge base lookup.
Given an input query, produce {{num_variants}} alternative phrasings that capture \
the same intent from different angles (e.g. more abstract, more specific, \
using synonyms, or from a different perspective).

Return exactly {{num_variants}} queries as a JSON object with a "variants" array.
"""


class _QueryVariants(BaseModel):
    variants: list[str]


class MultiQueryRetryExhaustedError(RetryExhaustedError):
    pass


class MultiQueryAgent(_BaseAgent):
    def __init__(self, llm: BaseChatModel, config: _LLMConfig, num_variants: int) -> None:
        super().__init__(llm, config)
        self._num_variants = num_variants

    @property
    def _system_prompt(self) -> str:
        return _MULTI_QUERY_PROMPT_TEMPLATE.replace("{{num_variants}}", str(self._num_variants))

    async def generate(self, query: str) -> list[str]:
        messages = [SystemMessage(self._system_prompt), HumanMessage(query)]
        result: _QueryVariants = await self._retryable_structured_ainvoke(
            messages,
            _QueryVariants,
            raise_on_exhaustion=MultiQueryRetryExhaustedError("multi-query exhausted retries"),
        )
        variants = result.variants[: self._num_variants]
        logger.debug("multi-query: generated %d variants for %r", len(variants), query[:60])
        return variants
