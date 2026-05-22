from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import ResolutionLLMConfig


class _CrossTypeOpenQuestionEntry(_EntryBase):
    rel_type: Literal[RelationshipType.BLOCKS] | None  # type: ignore[override]


class _CrossTypeOpenQuestionResult(_ResultBase[_CrossTypeOpenQuestionEntry]): ...


_OPEN_QUESTION_DECISION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating OpenQuestion → Decision pairs.

You receive a source open question (current meeting) and a list of target decisions (prior meeting KB).
For each target, determine whether the open question directly prevents the decision from being executed or enforced.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source open question → target decision only.
- Require a direct execution dependency — the question must need to be answered before the decision can be safely acted on.
- A question that is relevant to the decision's context without gating its execution does NOT block it.
- A question in the same domain that does not logically precede the decision does NOT block it.
- A question about final approval, acceptance, or governance status blocks only the decision whose finality it directly controls; it does not automatically block every downstream or related decision.
- A question that would refine implementation details is null if the decision can still be executed with reasonable defaults or documented assumptions.

Selection priority:
  1. Use blocks if the decision cannot be safely, meaningfully, or consistently executed until the question is answered.
  2. Otherwise, use null.

Relationship definitions:
- blocks: the source open question leaves a required prerequisite, parameter, constraint, or approval unresolved, so the target decision cannot be safely executed or treated as final.
    If the decision can proceed with an explicit assumption, temporary scope, or documented follow-up, use null.
    - Example: open_question "What cloud provider will we use?" blocks decision "Deploy the reporting warehouse on BigQuery" — BigQuery is provider-specific; the cloud provider choice must precede the deployment decision.
    - Example: open_question "What lifecycle policy applies to uploaded documents?" blocks decision "Enforce a 30-day deletion policy for uploaded support files" — the lifecycle policy question must be answered before a specific deletion policy can be set.
    - Example: open_question "Which compliance framework applies to EU tenant data?" blocks decision "Store EU tenant data in eu-central-1" — the regulatory requirements must be known before committing to a storage location.
- null: the open question does not gate the decision, or only shares domain context without a logical prerequisite relationship.
    - Example: open_question "Safety of five-minute exchange-rate cache for settlement calculations" and decision "Adopt RabbitMQ as the asynchronous job queue" — the cache question is unrelated to the queueing decision.
    - Example: open_question "What branching strategy should we use for hotfixes?" and decision "Require two-approver sign-off for production deployments" — same development workflow domain, no prerequisite dependency.
    - Example: open_question "Has the search architecture proposal received final acceptance?" and decision "Set index compatibility policy for search documents" — architecture acceptance is governance context, but the compatibility policy can be evaluated unless the acceptance question directly controls that policy.

Counter-examples:
- open_question "Should we use gRPC or REST for inter-service calls?" does NOT block decision "Add circuit breakers to all service calls" — circuit breakers are protocol-agnostic; the protocol question does not gate the circuit-breaker decision.
- open_question "What SLA can we commit to for the reporting API?" does NOT block decision "Set up read replicas for the analytics database" — the SLA question is downstream of the infrastructure decision, not a prerequisite for it.
- open_question "What logging format should we standardise on?" does NOT block decision "Deploy centralised log aggregation to production" — the deployment can proceed regardless of which format is standardised.
"""

_OPEN_QUESTION_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating OpenQuestion → ActionItem pairs.

You receive a source open question (current meeting) and a list of target action items (prior meeting KB).
For each target, determine whether the open question directly prevents the action item from being meaningfully executed.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source open question → target action item only.
- Require a direct execution dependency — the question must need to be answered before the action item can be started or completed in a meaningful way.
- A question that is relevant to the action item's context without gating its execution does NOT block it.
- A question that affects a different aspect of the same system does NOT block the action item.
- A question does NOT block an action item whose purpose is to answer, investigate, clarify, or gather input for that question.
- A question that only changes the final content of a deliverable is null if the action item can begin by documenting assumptions, options, or unresolved points.

Selection priority:
  1. Use blocks if the action item cannot be meaningfully started or completed until the question is answered — the question determines a key parameter, constraint, or direction for the task.
  2. Otherwise, use null.

Relationship definitions:
- blocks: the source open question withholds a required input, parameter, constraint, or direction needed to start or complete the target action item meaningfully.
    If the action item exists to answer, investigate, clarify, or gather input for the question, use null.
    - Example: open_question "What lifecycle policy applies to uploaded documents?" blocks action_item "Configure object-storage lifecycle settings" — the lifecycle settings are directly determined by the policy answer.
    - Example: open_question "Which cloud provider will we use?" blocks action_item "Provision the production Kubernetes cluster" — the cluster provisioning depends on the cloud provider choice.
    - Example: open_question "What GDPR data classification applies to user activity logs?" blocks action_item "Define access control policies for the activity log store" — the classification determines the required access controls.
- null: the open question does not gate the action item, or only shares domain context without a prerequisite dependency.
    - Example: open_question "Safety of five-minute exchange-rate cache for settlement calculations" and action_item "Billing Worker Scaling Proposal" — both touch caching but the proposal addresses throughput and can proceed independently.
    - Example: open_question "What branching strategy should we use for hotfixes?" and action_item "Set up CI/CD pipeline for the frontend" — same development workflow domain, no prerequisite dependency.
    - Example: open_question "What lifecycle exceptions are required for legal holds?" and action_item "Clarify uploaded-document lifecycle requirements with legal" — the action item is meant to answer the question, so the question does not block it.

Counter-examples:
- open_question "What SLA can we commit to for the reporting API?" does NOT block action_item "Add indexes to the reports table" — the indexing task can proceed regardless of the SLA answer; the SLA may depend on the indexes, not the other way around.
- open_question "Should we use gRPC or REST for inter-service calls?" does NOT block action_item "Write integration tests for the payments service" — the tests can be written against either protocol; the choice of protocol does not gate the testing task.
- open_question "What monitoring stack should we adopt?" does NOT block action_item "Add structured logging to the billing service" — structured logging is useful regardless of which monitoring stack is chosen.
"""

_PROMPTS: dict[ConceptType, str] = {
    ConceptType.DECISION: _OPEN_QUESTION_DECISION_PROMPT,
    ConceptType.ACTION_ITEM: _OPEN_QUESTION_ACTION_ITEM_PROMPT,
}


class OpenQuestionCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent[_CrossTypeOpenQuestionEntry]):
    """Resolves OpenQuestion → Decision (blocks), OpenQuestion → ActionItem (blocks)."""

    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig, target_type: ConceptType) -> None:
        super().__init__(llm=llm, config=config)
        self._target_type = target_type

    @property
    def _result_model(self) -> type[_CrossTypeOpenQuestionResult]:
        return _CrossTypeOpenQuestionResult

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
