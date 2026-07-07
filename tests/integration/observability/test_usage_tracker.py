"""Integration tests for UsageTracker wired through a real LLM call.

These tests verify that TokenBudgetCallback correctly intercepts usage_metadata
from a live LangChain structured-output call and accumulates token counts.
"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from seshat.app.agents.base import RetryExhaustedError, _BaseAgent
from seshat.app.platform.observability.usage_tracker import (
    TokenBudgetCallback,
    TokenBudgetExceededError,
    TrackingEmbeddings,
    UsageTracker,
    set_run_tracker,
    track_token_budget,
)
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API, SKIP_IF_NO_LLM_API

pytestmark = [pytest.mark.integration, pytest.mark.llm]


class _YesNo(BaseModel):
    answer: bool


class _MinimalAgent(_BaseAgent):
    @property
    def _system_prompt(self) -> str:
        return "Answer the question with yes or no."

    async def ask(self, question: str) -> _YesNo:
        messages = [SystemMessage(self._system_prompt), HumanMessage(question)]
        return await self._retryable_structured_ainvoke(
            messages,
            _YesNo,
            raise_on_exhaustion=RetryExhaustedError("exhausted"),
        )


class _AgentRunner:
    """Wraps _MinimalAgent in a track_token_budget-decorated method.

    job_tracker accumulates across repeated run() calls so tests can assert
    on total usage without inspecting the per-stage ContextVar directly.
    """

    def __init__(self, llm, config, *, max_input: int = 10_000, max_output: int = 1_000):
        self._llm = llm
        self._config = config
        self._max_input = max_input
        self._max_output = max_output
        self.job_tracker = UsageTracker.uncapped()

    @track_token_budget(
        max_input_fn=lambda self: self._max_input,
        max_output_fn=lambda self: self._max_output,
        label="test",
        accumulate_to_fn=lambda self: self.job_tracker,
    )
    async def run(self, question: str) -> _YesNo:
        return await _MinimalAgent(llm=self._llm, config=self._config).ask(question)


class TestUsageTrackerIntegration:
    @pytest.mark.asyncio
    @pytest.mark.agents
    @SKIP_IF_NO_LLM_API
    async def test_tokens_captured_via_callback(self, cheap_llm, identification_config):
        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=1_000)
        set_run_tracker(TokenBudgetCallback(tracker))

        agent = _MinimalAgent(llm=cheap_llm, config=identification_config)
        result = await agent.ask("Is Python a programming language?")

        assert result.answer is True
        assert tracker.input_tokens > 0
        assert tracker.output_tokens > 0

    @pytest.mark.asyncio
    @pytest.mark.agents
    @SKIP_IF_NO_LLM_API
    async def test_decorator_captures_tokens(self, cheap_llm, identification_config):
        runner = _AgentRunner(cheap_llm, identification_config)
        await runner.run("Is the sky blue?")

        assert runner.job_tracker.input_tokens > 0
        assert runner.job_tracker.output_tokens > 0

    @pytest.mark.asyncio
    @pytest.mark.agents
    @SKIP_IF_NO_LLM_API
    async def test_cap_exceeded_raises_after_real_call(self, cheap_llm, identification_config):
        runner = _AgentRunner(cheap_llm, identification_config, max_input=1, max_output=1)

        with pytest.raises(TokenBudgetExceededError):
            await runner.run("Is water wet?")

    @pytest.mark.asyncio
    @pytest.mark.agents
    @SKIP_IF_NO_LLM_API
    async def test_accumulate_to_fn_rolls_up_stage_totals(self, cheap_llm, identification_config):
        runner = _AgentRunner(cheap_llm, identification_config)

        await runner.run("Is the Earth round?")
        after_first = runner.job_tracker.input_tokens
        assert after_first > 0

        await runner.run("Is the Earth round?")
        assert runner.job_tracker.input_tokens > after_first

    @pytest.mark.asyncio
    @pytest.mark.embedding
    @SKIP_IF_NO_EMBEDDINGS_API
    async def test_embedding_tokens_captured(self, azure_embeddings: TrackingEmbeddings):
        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=1_000)
        set_run_tracker(TokenBudgetCallback(tracker))

        await azure_embeddings.aembed_query("The team agreed to use PostgreSQL.")

        assert tracker.embedding_input_tokens > 0
        assert tracker.input_tokens == 0  # embedding tokens are tracked separately from LLM input
        assert tracker.output_tokens == 0
