import pytest

from seshat.agents.identification.action_item import ActionItemIdentificationAgent
from seshat.agents.identification.base import AnchoredConcept
from seshat.agents.identification.decision import Decision, DecisionIdentificationAgent
from seshat.agents.identification.grouping import ConceptGroup, GroupingAgent
from seshat.agents.identification.open_question import OpenQuestionIdentificationAgent
from seshat.agents.identification.risk import RiskIdentificationAgent
from seshat.models.enums import ConceptType
from tests.integration.conftest import SKIP_IF_NO_LLM_API

pytestmark = [pytest.mark.integration, pytest.mark.agents, pytest.mark.llm, SKIP_IF_NO_LLM_API]

_TRANSCRIPT_FILE = "test_meeting.txt"


class TestDecisionIdentificationAgent:
    async def test_identify_returns_empty_for_non_extractable_transcript(self, cheap_llm, extraction_config):
        transcript = "The weather today is sunny. Everyone agrees it feels like spring."
        agent = DecisionIdentificationAgent(
            llm=cheap_llm,
            config=extraction_config.identification,
            grouped_identification_types=extraction_config.grouped_identification_types,
        )

        result = await agent.identify(transcript, kb_hint="", transcript_file=_TRANSCRIPT_FILE)

        assert result == []

    async def test_identify_finds_two_separate_decisions(self, cheap_llm, extraction_config):
        transcript = (
            "First, the team agreed to use PostgreSQL for the user database because of its JSONB support. "
            "Later, the team decided to deploy on AWS because the company already has a billing relationship there."
        )
        agent = DecisionIdentificationAgent(
            llm=cheap_llm,
            config=extraction_config.identification,
            grouped_identification_types=extraction_config.grouped_identification_types,
        )

        result = await agent.identify(transcript, kb_hint="", transcript_file=_TRANSCRIPT_FILE)

        total_members = sum(len(g.members) for g in result)
        assert total_members >= 2

    async def test_identify_finds_obvious_decision(self, cheap_llm, extraction_config):
        # Decision extraction goes through GroupingAgent (full pipeline) because
        # ConceptType.DECISION is in grouped_identification_types by default.
        transcript = (
            "The team reviewed the database options. PostgreSQL was proposed because of its native JSONB support, "
            "which the metadata store requires. MySQL was considered but ruled out because it lacks first-class JSON "
            "indexing. The team agreed to use PostgreSQL for the user database and closed the discussion."
        )
        agent = DecisionIdentificationAgent(
            llm=cheap_llm,
            config=extraction_config.identification,
            grouped_identification_types=extraction_config.grouped_identification_types,
        )

        result = await agent.identify(transcript, kb_hint="", transcript_file=_TRANSCRIPT_FILE)

        assert len(result) == 1
        assert isinstance(result[0], ConceptGroup)
        members = result[0].members
        assert len(members) == 1

        first = members[0].item
        assert first.decision
        assert first.title
        assert first.quote


class TestRiskIdentificationAgent:
    async def test_identify_finds_obvious_risk(self, cheap_llm, extraction_config):
        transcript = (
            "The migration window was discussed at length. Running the database migration during peak hours was raised "
            "as a serious concern — if a write fails mid-migration, partial data could be committed, leading to "
            "corruption that would be hard to detect without a full audit. The team agreed the failure mode was real "
            "and the consequences would be severe. The migration has not been scheduled yet and no window has been "
            "approved, so the risk of running it at the wrong time remains open."
        )
        agent = RiskIdentificationAgent(
            llm=cheap_llm,
            config=extraction_config.identification,
            grouped_identification_types=extraction_config.grouped_identification_types,
        )

        result = await agent.identify(transcript, kb_hint="", transcript_file=_TRANSCRIPT_FILE)

        assert len(result) == 1
        first = result[0]
        assert isinstance(first, AnchoredConcept)
        assert first.item.risk
        assert first.item.type in ("future", "blocker")
        assert first.item.title
        assert first.item.quote

    async def test_identify_with_kb_hint_still_finds_risk(self, cheap_llm, extraction_config):
        transcript = (
            "The migration window was discussed. Running the database migration during peak hours was raised "
            "as a concern — if a write fails mid-migration, partial data could be committed."
        )
        # Format matches _assemble_kb_hint: "<title> (date <iso>): <description[:80]>"
        kb_hint = (
            "Data loss during migration (date 2026-03-10): Risk of partial data commit if migration fails mid-run."
        )
        agent = RiskIdentificationAgent(
            llm=cheap_llm,
            config=extraction_config.identification,
            grouped_identification_types=extraction_config.grouped_identification_types,
        )

        result = await agent.identify(transcript, kb_hint=kb_hint, transcript_file=_TRANSCRIPT_FILE)

        assert len(result) >= 1
        assert all(isinstance(r, AnchoredConcept) for r in result)


class TestActionItemIdentificationAgent:
    async def test_identify_finds_obvious_action_item(self, cheap_llm, extraction_config):
        transcript = (
            "The migration script was discussed. It was agreed that someone needs to own the implementation. "
            "Sergio was asked to write the database migration script and deliver it by Friday. Sergio accepted "
            "the task and confirmed the Friday deadline was achievable."
        )
        agent = ActionItemIdentificationAgent(
            llm=cheap_llm,
            config=extraction_config.identification,
            grouped_identification_types=extraction_config.grouped_identification_types,
        )

        result = await agent.identify(transcript, kb_hint="", transcript_file=_TRANSCRIPT_FILE)

        assert len(result) == 1
        first = result[0]
        assert isinstance(first, AnchoredConcept)
        assert first.item.task
        assert first.item.assignee is not None
        assert first.item.title
        assert first.item.quote


class TestOpenQuestionIdentificationAgent:
    async def test_identify_finds_obvious_open_question(self, cheap_llm, extraction_config):
        transcript = (
            "The caching layer was discussed but no decision was reached. Redis and Memcached were both proposed. "
            "The team does not yet have enough load test data to evaluate which option handles the expected write "
            "throughput. The decision on which caching backend to adopt was deferred until the load tests are complete."
        )
        agent = OpenQuestionIdentificationAgent(
            llm=cheap_llm,
            config=extraction_config.identification,
            grouped_identification_types=extraction_config.grouped_identification_types,
        )

        result = await agent.identify(transcript, kb_hint="", transcript_file=_TRANSCRIPT_FILE)

        assert len(result) == 1
        first = result[0]
        assert isinstance(first, AnchoredConcept)
        assert first.item.question
        assert first.item.context
        assert first.item.title
        assert first.item.quote


class TestGroupingAgent:
    async def test_group_places_all_items_in_groups(self, cheap_llm, extraction_config):
        items = [
            AnchoredConcept(
                item=Decision(
                    quote="The team agreed to use PostgreSQL for the user database.",
                    title="Use PostgreSQL",
                    description="Team chose PostgreSQL for the user database",
                    decision="Use PostgreSQL for the user database.",
                    rationale="Better JSONB support needed for the metadata store.",
                    alternatives_considered=["MySQL"],
                ),
                quote_anchor=None,
            ),
            AnchoredConcept(
                item=Decision(
                    quote="The team will deploy on Kubernetes.",
                    title="Deploy on Kubernetes",
                    description="Team chose Kubernetes for container orchestration",
                    decision="Deploy on Kubernetes.",
                    rationale="Better scaling and self-healing capabilities.",
                    alternatives_considered=[],
                ),
                quote_anchor=None,
            ),
        ]
        agent = GroupingAgent(llm=cheap_llm, config=extraction_config.identification)

        groups = await agent.group(items, ConceptType.DECISION)

        assert len(groups) >= 1
        total_members = sum(len(g.members) for g in groups)
        assert total_members == 2
        assert all(g.group_title for g in groups)
