import pytest

from seshat.app.agents.resolution.cross_type.decision import DecisionCrossTypeResolutionAgent
from seshat.app.agents.resolution.same_type.action_item import ActionItemResolutionAgent
from seshat.app.agents.resolution.same_type.decision import DecisionResolutionAgent
from seshat.app.agents.resolution.same_type.open_question import OpenQuestionResolutionAgent
from seshat.app.agents.resolution.same_type.risk import RiskResolutionAgent
from seshat.core.models.enums import ConceptType, RelationshipType
from tests.helpers import make_node
from tests.integration.conftest import SKIP_IF_NO_LLM_API

pytestmark = [pytest.mark.integration, pytest.mark.agents, pytest.mark.llm, SKIP_IF_NO_LLM_API]


def _assert_valid_relationships(result, input_nodes, expected_rel_type: RelationshipType):
    assert len(result) >= 1
    valid_ids = {node.id for node in input_nodes}
    for item in result:
        assert item.source_id in valid_ids
        assert item.target_id in valid_ids
        assert isinstance(item.rel_type, RelationshipType)
        assert item.rationale
    assert any(item.rel_type == expected_rel_type for item in result)


class TestDecisionResolutionAgent:
    async def test_resolve_supersedes_old_decision(self, cheap_llm, resolution_config):
        old_decision = make_node(
            node_id="decision-old",
            type=ConceptType.DECISION,
            title="Use PostgreSQL v12",
            description="The team decided to use PostgreSQL v12 for the user database.",
        )
        new_decision = make_node(
            node_id="decision-new",
            type=ConceptType.DECISION,
            title="Use PostgreSQL v15 (supersedes v12 due to improved performance)",
            description=(
                "The team decided to upgrade to PostgreSQL v15, replacing the earlier v12 decision "
                "due to significant performance improvements and better JSON support in v15."
            ),
        )
        agent = DecisionResolutionAgent(llm=cheap_llm, config=resolution_config)

        result, _ = await agent.resolve(
            source_nodes=[new_decision],
            per_source_targets={new_decision.id: [old_decision]},
        )

        _assert_valid_relationships(result, [new_decision, old_decision], RelationshipType.SUPERSEDES)

    async def test_resolve_returns_empty_for_unrelated_decisions(self, cheap_llm, resolution_config):
        decision_a = make_node(
            node_id="decision-unrelated-a",
            type=ConceptType.DECISION,
            title="Use Terraform for infrastructure provisioning",
            description="The team decided to adopt Terraform to manage all cloud infrastructure as code.",
        )
        decision_b = make_node(
            node_id="decision-unrelated-b",
            type=ConceptType.DECISION,
            title="Use Figma for UI design collaboration",
            description="The design team will use Figma as the single tool for mockups and design handoff.",
        )
        agent = DecisionResolutionAgent(llm=cheap_llm, config=resolution_config)

        result, failed = await agent.resolve(
            source_nodes=[decision_a],
            per_source_targets={decision_a.id: [decision_b]},
        )

        assert result == []
        assert failed == []


class TestRiskResolutionAgent:
    async def test_resolve_amended_risk(self, cheap_llm, resolution_config):
        original_risk = make_node(
            node_id="risk-original",
            type=ConceptType.RISK,
            title="Database migration may corrupt data",
            description=(
                "Running the database migration without a rehearsal increases the risk of data corruption "
                "if the migration script contains errors."
            ),
        )
        amended_risk = make_node(
            node_id="risk-amended",
            type=ConceptType.RISK,
            title="Database migration risk is mitigated by dry-run procedure",
            description=(
                "The original data corruption risk during migration is now qualified: a mandatory dry-run "
                "procedure has been added, reducing but not eliminating the corruption risk."
            ),
        )
        agent = RiskResolutionAgent(llm=cheap_llm, config=resolution_config)

        result, _ = await agent.resolve(
            source_nodes=[amended_risk],
            per_source_targets={amended_risk.id: [original_risk]},
        )

        _assert_valid_relationships(result, [amended_risk, original_risk], RelationshipType.AMENDS)


class TestActionItemResolutionAgent:
    async def test_resolve_blocked_action_item(self, cheap_llm, resolution_config):
        blocker = make_node(
            node_id="action-blocker",
            type=ConceptType.ACTION_ITEM,
            title="Write migration script",
            description=(
                "Sergio must write the database migration script before any deployment can proceed. "
                "This task is a prerequisite for the production deployment."
            ),
        )
        blocked = make_node(
            node_id="action-blocked",
            type=ConceptType.ACTION_ITEM,
            title="Deploy to production",
            description=(
                "Deploy the new service to production. This cannot proceed until the migration script "
                "is written and reviewed."
            ),
        )
        agent = ActionItemResolutionAgent(llm=cheap_llm, config=resolution_config)

        result, _ = await agent.resolve(
            source_nodes=[blocker],
            per_source_targets={blocker.id: [blocked]},
        )

        _assert_valid_relationships(result, [blocker, blocked], RelationshipType.BLOCKS)


class TestOpenQuestionResolutionAgent:
    async def test_resolve_dependent_open_questions(self, cheap_llm, resolution_config):
        prerequisite_question = make_node(
            node_id="oq-prereq",
            type=ConceptType.OPEN_QUESTION,
            title="Which cloud provider will we use?",
            description=(
                "The team has not yet decided on a cloud provider. This decision affects all "
                "infrastructure choices downstream."
            ),
        )
        dependent_question = make_node(
            node_id="oq-dependent",
            type=ConceptType.OPEN_QUESTION,
            title="Which managed database service should we use?",
            description=(
                "The choice of managed database service depends on which cloud provider is selected. "
                "This question cannot be answered until the cloud provider question is resolved."
            ),
        )
        agent = OpenQuestionResolutionAgent(llm=cheap_llm, config=resolution_config)

        result, _ = await agent.resolve(
            source_nodes=[dependent_question],
            per_source_targets={dependent_question.id: [prerequisite_question]},
        )

        _assert_valid_relationships(result, [dependent_question, prerequisite_question], RelationshipType.DEPENDS_ON)


class TestDecisionCrossTypeResolutionAgent:
    async def test_resolve_decision_mitigates_risk(self, cheap_llm, resolution_config):
        decision = make_node(
            node_id="cross-decision",
            type=ConceptType.DECISION,
            title="Migrate during maintenance window",
            description=(
                "The team decided to perform the database migration exclusively during the scheduled "
                "maintenance window (Sunday 02:00-04:00 UTC) to minimize exposure during peak load "
                "and reduce the blast radius of any data corruption event."
            ),
        )
        risk = make_node(
            node_id="cross-risk",
            type=ConceptType.RISK,
            title="Data corruption risk during migration",
            description=(
                "Running the database migration during peak hours increases the probability of partial "
                "writes and data corruption if the migration script fails mid-execution."
            ),
        )
        agent = DecisionCrossTypeResolutionAgent(
            llm=cheap_llm,
            config=resolution_config,
            target_type=ConceptType.RISK,
        )

        result, _ = await agent.resolve(
            source_nodes=[decision],
            per_source_targets={decision.id: [risk]},
        )

        _assert_valid_relationships(result, [decision, risk], RelationshipType.MITIGATES)
