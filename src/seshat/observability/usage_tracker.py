from __future__ import annotations

import asyncio
import functools
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult

from seshat.observability.usage_logger import log_token_metrics
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

logger = get_logger(__name__)

_run_tracker_var: ContextVar[TokenBudgetCallback | None] = ContextVar("run_tracker", default=None)

_WARN_THRESHOLD = 0.9
# Concurrent agents may exceed the cap before any single task can observe it.
# Allow up to 10% overage before raising to avoid aborting runs that are only marginally over.
_RAISE_THRESHOLD = 1.1


class TokenBudgetExceededError(Exception):
    pass


class UsageTracker:
    _UNCAPPED = 2**63 - 1  # sentinel for job-level trackers that never raise

    def __init__(self, max_input_tokens: int, max_output_tokens: int) -> None:
        self._max_input = max_input_tokens
        self._max_output = max_output_tokens
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0
        self._lock = asyncio.Lock()

    @classmethod
    def uncapped(cls) -> UsageTracker:
        return cls(max_input_tokens=cls._UNCAPPED, max_output_tokens=cls._UNCAPPED)

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def cache_read_tokens(self) -> int:
        return self._cache_read_tokens

    @property
    def cache_creation_tokens(self) -> int:
        return self._cache_creation_tokens

    async def add(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        async with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cache_read_tokens += cache_read_tokens
            self._cache_creation_tokens += cache_creation_tokens

        in_pct = self._input_tokens / self._max_input
        out_pct = self._output_tokens / self._max_output
        if in_pct >= _WARN_THRESHOLD or out_pct >= _WARN_THRESHOLD:
            logger.warning(
                "Token budget at %.0f%% input / %.0f%% output (%d/%d in, %d/%d out)",
                in_pct * 100,
                out_pct * 100,
                self._input_tokens,
                self._max_input,
                self._output_tokens,
                self._max_output,
            )

    def check_caps(self) -> None:
        """Raise TokenBudgetExceededError if either cap has been exceeded by more than _RAISE_THRESHOLD."""
        if self._input_tokens > self._max_input * _RAISE_THRESHOLD:
            raise TokenBudgetExceededError(f"Input token cap exceeded: {self._input_tokens} > {self._max_input}")
        if self._output_tokens > self._max_output * _RAISE_THRESHOLD:
            raise TokenBudgetExceededError(f"Output token cap exceeded: {self._output_tokens} > {self._max_output}")

    def log_totals(self, label: str) -> None:
        logger.info(
            "%s token usage: input=%d/%d (%.1f%%), output=%d/%d (%.1f%%)",
            label,
            self._input_tokens,
            self._max_input,
            self._input_tokens / self._max_input * 100,
            self._output_tokens,
            self._max_output,
            self._output_tokens / self._max_output * 100,
        )


class TokenBudgetCallback(AsyncCallbackHandler):
    def __init__(self, tracker: UsageTracker) -> None:
        self._tracker = tracker

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                if isinstance(gen, ChatGeneration) and gen.message.usage_metadata:
                    usage = gen.message.usage_metadata
                    details = usage.get("input_token_details", {})
                    await self._tracker.add(
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        cache_read_tokens=details.get("cache_read", 0),
                        cache_creation_tokens=details.get("cache_creation", 0),
                    )


def set_run_tracker(callback: TokenBudgetCallback) -> None:
    # Set inside the orchestrator coroutine before spawning tasks — child tasks inherit
    # the ContextVar value (same callback object) so all concurrent agents accumulate
    # into the same tracker without needing any signature changes.
    _run_tracker_var.set(callback)


def get_run_tracker() -> TokenBudgetCallback | None:
    return _run_tracker_var.get()


def track_token_budget(
    max_input_fn: Callable[[Any], int],
    max_output_fn: Callable[[Any], int],
    label: str,
    accumulate_to_fn: Callable[[Any], UsageTracker] | None = None,
) -> Callable:
    """Decorator for async instance methods: creates a per-call UsageTracker, sets it on the
    ContextVar so all LLM calls within the method accumulate into it, checks caps on completion,
    and logs totals. Caps are read from the instance at call time via max_input_fn(self) /
    max_output_fn(self) so config changes are always respected.

    If accumulate_to_fn is provided, stage totals are rolled up into that tracker (typically a
    job-level uncapped tracker on self) so callers can surface per-job usage without re-running."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            tracker = UsageTracker(max_input_fn(self), max_output_fn(self))
            set_run_tracker(TokenBudgetCallback(tracker))
            try:
                result = await fn(self, *args, **kwargs)
                tracker.check_caps()
                return result
            finally:
                tracker.log_totals(label)
                log_token_metrics(
                    label,
                    input_tokens=tracker.input_tokens,
                    output_tokens=tracker.output_tokens,
                    cache_read_tokens=tracker.cache_read_tokens,
                    cache_creation_tokens=tracker.cache_creation_tokens,
                )

                if accumulate_to_fn is not None:
                    await accumulate_to_fn(self).add(
                        input_tokens=tracker.input_tokens,
                        output_tokens=tracker.output_tokens,
                        cache_read_tokens=tracker.cache_read_tokens,
                        cache_creation_tokens=tracker.cache_creation_tokens,
                    )

        return wrapper

    return decorator
