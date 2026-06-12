"""Tests for UsageTracker, TokenBudgetCallback, and track_token_budget."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from seshat.observability.usage_tracker import (
    TokenBudgetCallback,
    TokenBudgetExceededError,
    UsageTracker,
    get_run_tracker,
    track_token_budget,
)


def _result(
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> LLMResult:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestUsageTracker:
    @pytest.mark.asyncio
    async def test_accumulates_tokens(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        await tracker.add(input_tokens=100, output_tokens=50)
        await tracker.add(input_tokens=200, output_tokens=80)
        assert tracker.input_tokens == 300
        assert tracker.output_tokens == 130

    @pytest.mark.asyncio
    async def test_accumulates_cache_tokens(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        await tracker.add(input_tokens=0, output_tokens=0, cache_read_tokens=40, cache_creation_tokens=10)
        await tracker.add(input_tokens=0, output_tokens=0, cache_read_tokens=60, cache_creation_tokens=5)
        assert tracker.cache_read_tokens == 100
        assert tracker.cache_creation_tokens == 15

    def test_check_caps_passes_within_limits(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        tracker._input_tokens = 1050  # within 10% overage
        tracker._output_tokens = 525
        tracker.check_caps()  # should not raise

    def test_check_caps_raises_on_input_exceeded(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        tracker._input_tokens = 1101  # over 110%
        with pytest.raises(TokenBudgetExceededError, match="Input token cap exceeded"):
            tracker.check_caps()

    def test_check_caps_raises_on_output_exceeded(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        tracker._output_tokens = 551  # over 110%
        with pytest.raises(TokenBudgetExceededError, match="Output token cap exceeded"):
            tracker.check_caps()

    @pytest.mark.asyncio
    async def test_accumulates_embedding_tokens(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        await tracker.add(embedding_input_tokens=120)
        await tracker.add(embedding_input_tokens=80)
        assert tracker.embedding_input_tokens == 200
        assert tracker.input_tokens == 0

    def test_check_caps_raises_on_embedding_exceeded(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500, max_embedding_tokens=100)
        tracker._embedding_input_tokens = 111  # over 110%
        with pytest.raises(TokenBudgetExceededError, match="Embedding token cap exceeded"):
            tracker.check_caps()


class TestTokenBudgetCallback:
    @pytest.mark.asyncio
    async def test_accumulates_from_llm_result(self):
        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=10_000)
        callback = TokenBudgetCallback(tracker)

        await callback.on_llm_end(_result(42, 17), run_id=uuid4())

        assert tracker.input_tokens == 42
        assert tracker.output_tokens == 17

    @pytest.mark.asyncio
    async def test_accumulates_cache_tokens_from_llm_result(self):
        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=10_000)
        callback = TokenBudgetCallback(tracker)

        await callback.on_llm_end(_result(10, 5, cache_read=30, cache_creation=20), run_id=uuid4())

        assert tracker.cache_read_tokens == 30
        assert tracker.cache_creation_tokens == 20

    @pytest.mark.asyncio
    async def test_skips_generation_without_usage_metadata(self):
        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=10_000)
        callback = TokenBudgetCallback(tracker)

        result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]])
        await callback.on_llm_end(result, run_id=uuid4())

        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0


class TestTrackTokenBudget:
    @pytest.mark.asyncio
    async def test_sets_run_tracker_before_fn(self):
        captured = []

        class _Obj:
            @track_token_budget(lambda self: 1000, lambda self: 500, label="test")
            async def run(self):
                captured.append(get_run_tracker())

        await _Obj().run()
        assert len(captured) == 1
        assert captured[0] is not None

    @pytest.mark.asyncio
    async def test_raises_on_cap_exceeded(self):
        class _Obj:
            @track_token_budget(lambda self: 10, lambda self: 10, label="test")
            async def run(self):
                cb = get_run_tracker()
                assert cb is not None
                await cb._tracker.add(input_tokens=200, output_tokens=0)  # well over 110%

        with pytest.raises(TokenBudgetExceededError):
            await _Obj().run()

    @pytest.mark.asyncio
    async def test_does_not_raise_within_overage_allowance(self):
        class _Obj:
            @track_token_budget(lambda self: 1000, lambda self: 500, label="test")
            async def run(self):
                cb = get_run_tracker()
                assert cb is not None
                await cb._tracker.add(input_tokens=1050, output_tokens=0)  # within 10% overage

        await _Obj().run()  # should not raise

    @pytest.mark.asyncio
    async def test_tracker_isolated_per_call(self):
        trackers = []

        class _Obj:
            @track_token_budget(lambda self: 10_000, lambda self: 10_000, label="test")
            async def run(self):
                trackers.append(get_run_tracker())

        obj = _Obj()
        await obj.run()
        await obj.run()

        assert len(trackers) == 2
        assert trackers[0] is not trackers[1]
