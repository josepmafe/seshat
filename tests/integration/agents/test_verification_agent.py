import pytest

from seshat.agents.verification import VerificationAgent
from tests.integration.conftest import SKIP_IF_NO_LLM_API

pytestmark = [pytest.mark.integration, pytest.mark.agents, pytest.mark.llm, SKIP_IF_NO_LLM_API]


class TestVerificationAgent:
    async def test_verify_returns_supported_true_for_matching_quote(self, cheap_llm, verification_config):
        agent = VerificationAgent(llm=cheap_llm, config=verification_config)

        result = await agent.verify(
            title="Use PostgreSQL",
            description="The team agreed to go with PostgreSQL.",
            quote="Agreed. Let's go with PostgreSQL.",
        )

        assert result.supported is True

    async def test_verify_returns_supported_false_for_contradicting_quote(self, cheap_llm, verification_config):
        agent = VerificationAgent(llm=cheap_llm, config=verification_config)

        result = await agent.verify(
            title="Use PostgreSQL",
            description="The team decided to use PostgreSQL for the user database.",
            quote="We evaluated both options and agreed MySQL is the better fit for our workload.",
        )

        assert result.supported is False
