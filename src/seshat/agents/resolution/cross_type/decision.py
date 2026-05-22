from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import ResolutionLLMConfig


class _CrossTypeDecisionEntry(_EntryBase):
    rel_type: Literal[RelationshipType.MITIGATES, RelationshipType.RESOLVES, RelationshipType.BLOCKS] | None  # type: ignore[override]


class _CrossTypeDecisionResult(_ResultBase[_CrossTypeDecisionEntry]): ...


_DECISION_RISK_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → Risk pairs.

You receive a source decision (current meeting) and a list of target risks (prior meeting KB).
For each target, determine whether the decision directly addresses the concern that motivated the risk.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source decision → target risk only.
- Require direct causal coupling — the decision must address the specific failure mode, not merely the same domain.
- A decision that reduces risk incidentally or addresses a related but distinct failure mode does NOT qualify.
- A decision that only detects, monitors, dashboards, or alerts on a risk does NOT mitigate the underlying technical or operational failure mode.
- Detection, monitoring, dashboards, or alerts qualify as mitigates only when the target risk's stated failure mode is missing detection, missing alerting, or missing incident response.
- A decision that merely acknowledges, defers, or postpones work does NOT mitigate a risk unless it introduces a concrete control that reduces likelihood, severity, or exposure.

Selection priority:
  1. Use mitigates if the decision directly removes, reduces, contains, or controls the failure mode or concern that motivated the risk.
  2. Otherwise, use null.

Relationship definitions:
- mitigates: the source decision establishes a policy, architecture, constraint, or control that directly reduces the target risk's likelihood, severity, exposure, or blast radius.
    The decision must mechanistically reduce the risk's failure mode; it must not merely be in the same domain or improve awareness of the risk.
    - Example: decision "Add dead-letter queue for failed events" mitigates risk "Silent event loss if consumer crashes".
    - Example: decision "Enable mutual TLS for all inter-service calls" mitigates risk "Inter-service traffic may be intercepted by a compromised internal node".
    - Example: decision "Set hard memory limits per image-processing worker" mitigates risk "Image-processing workers may exhaust available memory on oversized uploads".
- null: the decision does not directly address the risk's failure mode, or shares only a domain.
    - Example: decision "Use PostgreSQL for all storage" and risk "API gateway becomes unavailable under high load" — unrelated concerns.
    - Example: decision "Adopt statistical fraud scoring" and risk "Invoice export service degrades above 5 000 writes/min" — same analytics domain, unrelated mechanisms.
    - Example: decision "Add latency alerts for the billing API" and risk "Billing calculation workers will miss processing deadlines if invoice volume doubles" — alerts detect the breach but do not reduce the worker load causing it.

Counter-examples:
- decision "Add circuit breakers to all service calls" does NOT mitigate risk "Database write-ahead log may fill up during a bulk import" — circuit breakers address service availability, not log capacity.
- decision "Enforce RBAC for the admin panel" does NOT mitigate risk "API rate limiting may allow a single client to exhaust capacity" — both are protection concerns but different failure modes.
- decision "Adopt blue-green deployments" does NOT mitigate risk "Cache warming failure causes elevated latency after a deployment" — the deployment strategy does not address the cache warming mechanism.
- decision "Defer automated incident classification until operations ownership is clarified" does NOT mitigate risk "Automated incident classification lacks owner-defined escalation criteria" — deferral avoids immediate commitment but does not reduce the underlying ownership risk.
"""

_DECISION_OPEN_QUESTION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → OpenQuestion pairs.

You receive a source decision (current meeting) and a list of target open questions (prior meeting KB).
For each target, determine whether the decision directly and fully resolves the open question.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source decision → target open question only.
- Only assign resolves if the decision fully and directly settles the question — partial or tangential answers do not qualify.
- A decision that answers only a subcase or instance of the question does NOT resolve it; it may instead narrow it.
- A decision that narrows or adds constraints to the question without answering it is null, not resolves.
- A phased, provisional, experimental, or temporary decision does NOT resolve a broader open question unless the question is explicitly scoped to that phase.
- A decision that assumes or depends on an answer to the question does NOT resolve the question.

Selection priority:
  1. Use resolves if the decision provides a direct, complete answer that closes the open question.
  2. Otherwise, use null.

Relationship definitions:
- resolves: the source decision gives a direct, complete, and final-enough answer to the target open question, so the question should no longer be tracked as open.
    Partial answers, phase-one choices, assumptions, or decisions that only narrow the answer space do not resolve the question.
    - Example: decision "Use blue/green deployments for all production releases" resolves open_question "What deployment strategy should we use?".
    - Example: decision "Store all EU tenant data in eu-central-1" resolves open_question "Where should EU tenant data be hosted?".
    - Example: decision "Adopt RabbitMQ as the asynchronous job queue" resolves open_question "What queueing technology should background jobs use?".
- null: the decision does not fully resolve the open question, or only partially addresses it.
    - Example: decision "Cap fraud-score cache TTL to 2 min for high-risk transactions" and open_question "What is the overall customer data handling policy?" — the decision narrows one aspect but does not settle the broader question.
    - Example: decision "Enable mutual TLS for all inter-service calls" and open_question "What authentication model should we use for external APIs?" — related security domain, different scope.
    - Example: decision "Build a phase-one automated ticket router" and open_question "Should we build or buy the long-term ticket routing capability?" — the decision settles an initial phase but not the broader long-term choice.

Counter-examples:
- decision "Enforce two-approver sign-off for production deployments" does NOT resolve open_question "How should we handle emergency hotfix deployments?" — it sets a general policy but does not specifically settle the hotfix exception.
- decision "Use Redis for session storage" does NOT resolve open_question "What caching strategy should we adopt across all services?" — the question is broader than the decision's scope.
- decision "Deploy the reporting warehouse on BigQuery" does NOT resolve open_question "What cloud provider will we use?" — the warehouse decision presupposes a cloud provider but does not itself answer the provider question.
"""

_DECISION_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → ActionItem pairs.

You receive a source decision (current meeting) and a list of target action items (prior meeting KB).
For each target, determine whether the decision directly hinders or prevents the action item from being executed.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source decision → target action item only.
- Require a direct execution dependency — the decision must make the specific action item impossible or impractical to complete as stated.
- A decision that changes the context of an action item without stopping it does NOT block it.
- A decision that provides a constraint, input, or parameter for an action item does NOT block it if the action item can proceed by incorporating that decision.
- A decision that makes an action item redundant or superseded is NOT blocks — those are same-type relationships.

Selection priority:
  1. Use blocks if the decision imposes a constraint, restriction, or freeze that prevents the action item from proceeding as stated.
  2. Otherwise, use null.

Relationship definitions:
- blocks: the source decision imposes a restriction, freeze, prohibition, or incompatible constraint that prevents the target action item from proceeding as stated.
    If the action item can proceed by incorporating the decision as a constraint or updating its deliverable, use null.
    - Example: decision "Defer all non-critical deployments pending security review" blocks action_item "Deploy new billing service to production".
    - Example: decision "Freeze all schema changes until migration is complete" blocks action_item "Add nullable audit_ts column to events table".
    - Example: decision "Require legal sign-off before any PII export" blocks action_item "Export historical user activity logs for the data science team".
- null: the decision does not prevent the action item from proceeding, or the relationship is only contextual.
    - Example: decision "Adopt statistical fraud scoring for phase one" and action_item "Sync on updated metric names for the support dashboard" — same monitoring domain but the metric sync can proceed regardless.
    - Example: decision "Use PostgreSQL for all storage" and action_item "Write onboarding documentation for new engineers" — unrelated concerns.
    - Example: decision "Search indexes must not be used as the system of record" and action_item "Write the search storage design note" — the design note can proceed by documenting that constraint; the decision changes required content but does not prevent the task.

Counter-examples:
- decision "Adopt Redis for session storage" does NOT block action_item "Implement OAuth provider integration" — the session storage choice is a constraint for the OAuth implementation but does not prevent the work from starting.
- decision "Enable mutual TLS for all inter-service calls" does NOT block action_item "Set up CI/CD pipeline for the frontend" — they share infrastructure concerns but the CI/CD pipeline setup is independent.
- decision "Cap exchange-rate cache TTL to 2 min for settlement calculations" does NOT block action_item "Benchmark cache hit rates under peak load" — the decision changes the TTL parameter but does not prevent the benchmarking task.
"""

_PROMPTS: dict[ConceptType, str] = {
    ConceptType.RISK: _DECISION_RISK_PROMPT,
    ConceptType.OPEN_QUESTION: _DECISION_OPEN_QUESTION_PROMPT,
    ConceptType.ACTION_ITEM: _DECISION_ACTION_ITEM_PROMPT,
}


class DecisionCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent[_CrossTypeDecisionEntry]):
    """Resolves Decision → Risk (mitigates), Decision → OpenQuestion (resolves), Decision → ActionItem (blocks)."""

    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig, target_type: ConceptType) -> None:
        super().__init__(llm=llm, config=config)
        self._target_type = target_type

    @property
    def _result_model(self) -> type[_CrossTypeDecisionResult]:
        return _CrossTypeDecisionResult

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
