from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.agents.identification.base import (
    AnchoredConcept,
    IdentificationRetryExhaustedError,
    _BaseIdentificationAgent,
)
from seshat.agents.identification.decision import Decision, DecisionList
from seshat.config.settings import IdentificationLLMConfig
from seshat.models.enums import ConceptType
from tests.helpers import make_structured_llm


class ConcreteAgent(_BaseIdentificationAgent):
    @property
    def concept_type(self) -> ConceptType:
        return ConceptType.DECISION

    @property
    def output_schema(self):
        return DecisionList

    @property
    def _system_prompt(self) -> str:
        return "You are a test agent."


def _make_decision(quote: str = "transcript text") -> Decision:
    return Decision(
        quote=quote,
        title="Use PostgreSQL",
        description="Team decided to use PostgreSQL.",
        decision="Use PostgreSQL for all data storage.",
        rationale="Not stated.",
    )


def _make_agent(llm, grouped_identification_types=frozenset(), **llm_kwargs) -> ConcreteAgent:
    return ConcreteAgent(
        llm=llm,
        config=IdentificationLLMConfig(**llm_kwargs),
        grouped_identification_types=set(grouped_identification_types),
    )


class TestBaseIdentificationAgent:
    @pytest.mark.asyncio
    async def test_identify_returns_anchored_concepts_on_success(self):
        item = _make_decision()
        llm = make_structured_llm(return_value=DecisionList(items=[item]))

        agent = _make_agent(llm)
        results = await agent.identify(transcript="transcript text", kb_hint="", transcript_file="test.txt")
        assert len(results) == 1
        assert results[0].item.title == "Use PostgreSQL"

    @pytest.mark.asyncio
    async def test_identify_raises_on_all_retries_fail(self):
        llm = make_structured_llm(side_effect=Exception("LLM error"))

        agent = _make_agent(llm, max_retries=2)
        with pytest.raises(IdentificationRetryExhaustedError):
            await agent.identify(transcript="text", kb_hint="", transcript_file="test.txt")

    @pytest.mark.asyncio
    async def test_fuzzy_quote_set_when_verbatim_match(self):
        item = _make_decision(quote="transcript text")
        llm = make_structured_llm(return_value=DecisionList(items=[item]))

        agent = _make_agent(llm)
        results = await agent.identify(transcript="transcript text", kb_hint="", transcript_file="test.txt")
        assert results[0].quote_anchor is not None
        assert results[0].quote_anchor.char_start == 0
        assert results[0].quote_anchor.char_end == len("transcript text")

    @pytest.mark.asyncio
    async def test_fuzzy_quote_none_when_no_match(self):
        item = _make_decision(quote="something not in transcript")
        llm = make_structured_llm(return_value=DecisionList(items=[item]))

        agent = _make_agent(llm)
        results = await agent.identify(transcript="transcript text", kb_hint="", transcript_file="test.txt")
        assert results[0].quote_anchor is None

    @pytest.mark.asyncio
    async def test_identify_routes_through_grouping_when_type_in_config(self):
        from seshat.agents.identification.grouping import ConceptGroup, _GroupingSchema, _GroupSchema

        item = _make_decision()
        identification_llm = MagicMock()
        identification_llm.ainvoke = AsyncMock(return_value=DecisionList(items=[item]))

        grouping_schema = _GroupingSchema(
            groups=[
                _GroupSchema(group_title="Storage", group_description="DB decisions", member_ids=["D01"]),
            ]
        )
        grouping_llm = MagicMock()
        grouping_llm.ainvoke = AsyncMock(return_value=grouping_schema)

        llm = MagicMock()
        llm.with_structured_output = MagicMock(side_effect=[identification_llm, grouping_llm])

        agent = _make_agent(llm, grouped_identification_types={ConceptType.DECISION})
        results = await agent.identify(transcript="transcript text", kb_hint="", transcript_file="test.txt")

        assert len(results) == 1
        assert isinstance(results[0], ConceptGroup)
        assert results[0].group_title == "Storage"

    @pytest.mark.asyncio
    async def test_identify_skips_grouping_when_type_not_in_config(self):
        item = _make_decision()
        llm = make_structured_llm(return_value=DecisionList(items=[item]))

        agent = _make_agent(llm)
        results = await agent.identify(transcript="transcript text", kb_hint="", transcript_file="test.txt")

        assert len(results) == 1
        assert isinstance(results[0], AnchoredConcept)
        assert llm.with_structured_output.call_count == 1
