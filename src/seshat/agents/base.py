from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from seshat.utils.hashing import fingerprint
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import _LLMConfig

M = TypeVar("M", bound=BaseModel)


logger = get_logger(__name__)


class RetryExhaustedError(Exception):
    pass


class _BaseAgent:
    """Base class for all LLM-calling agents. Provides a structured-output call with exponential backoff retry."""

    def __init__(self, llm: BaseChatModel, config: _LLMConfig) -> None:
        self._llm = llm
        self._max_retries = config.max_retries

    def fingerprint(self) -> str:
        return fingerprint(self._system_prompt)

    def prompt_texts(self) -> dict[str, str]:
        return {"system": self._system_prompt}

    @property
    def _system_prompt(self) -> str:
        raise NotImplementedError

    async def _retryable_structured_ainvoke(
        self,
        messages: list,
        response_model: type[M],
        *,
        raise_on_exhaustion: RetryExhaustedError,
        on_error_log_prefix: str | None = None,
    ) -> M:
        structured = self._llm.with_structured_output(response_model)
        on_error_log_prefix = on_error_log_prefix or response_model.__name__
        attempts = max(1, self._max_retries)
        for attempt in range(attempts):
            try:
                result = await structured.ainvoke(messages)
            except Exception as exc:
                delay = 0.5 * (2**attempt) + random.uniform(0, 0.1)
                logger.warning(
                    "%s attempt %d/%d failed — retrying in %.2fs due to %s: %s",
                    on_error_log_prefix,
                    attempt + 1,
                    attempts,
                    delay,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)
            else:
                assert_msg = f"Expected LLM output to be {response_model.__name__}, got {type(result).__name__}"
                assert isinstance(result, response_model), assert_msg
                return result

        raise raise_on_exhaustion
