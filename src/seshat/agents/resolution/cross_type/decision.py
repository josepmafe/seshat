from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import ResolutionLLMConfig


class _DecisionToRiskEntry(_EntryBase):
    rel_type: Literal[RelationshipType.MITIGATES] | None  # type: ignore[override]


class _DecisionToOpenQuestionEntry(_EntryBase):
    rel_type: Literal[RelationshipType.RESOLVES] | None  # type: ignore[override]


class _DecisionToActionItemEntry(_EntryBase):
    rel_type: Literal[RelationshipType.BLOCKS] | None  # type: ignore[override]


class _DecisionToRiskResult(_ResultBase[_DecisionToRiskEntry]): ...


class _DecisionToOpenQuestionResult(_ResultBase[_DecisionToOpenQuestionEntry]): ...


class _DecisionToActionItemResult(_ResultBase[_DecisionToActionItemEntry]): ...


_DecisionResult = _DecisionToRiskResult | _DecisionToOpenQuestionResult | _DecisionToActionItemResult


_DECISION_RISK_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → Risk pairs.

You receive a source decision (current meeting) and a list of target risks (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source decision → target risk only.

Not mitigates:
- The decision and risk share a domain but the decision addresses a different failure mode.
  Counter-example: "Add circuit breakers to all service calls" is NOT mitigates for "Database write-ahead log may fill up during a bulk import" — circuit breakers address service availability, not log capacity → null.
- The decision only detects, monitors, or alerts on the risk without reducing the failure mode itself.
  Counter-example: "Add latency alerts for the billing API" is NOT mitigates for "Billing calculation workers will miss processing deadlines if invoice volume doubles" — alerts detect the breach but do not reduce the worker load → null.
- The decision defers, acknowledges, or postpones work without introducing a concrete control.
  Counter-example: "Defer automated incident classification until operations ownership is clarified" is NOT mitigates for "Automated incident classification lacks owner-defined escalation criteria" — deferral does not reduce the underlying risk → null.

mitigates — when to use it:
The decision establishes a policy, architecture, constraint, or control that directly reduces the risk's likelihood, severity, exposure, or blast radius. The mechanism must directly address the risk's specific failure mode.
- Example: "Add dead-letter queue for failed events" mitigates "Silent event loss if consumer crashes".
- Example: "Enable mutual TLS for all inter-service calls" mitigates "Inter-service traffic may be intercepted by a compromised internal node".
- Example: "Set hard memory limits per image-processing worker" mitigates "Image-processing workers may exhaust available memory on oversized uploads".

Selection:
  1. Use mitigates if the decision directly removes, reduces, contains, or controls the specific failure mode that motivated the risk.
  2. Otherwise, use null.
"""

_DECISION_OPEN_QUESTION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → OpenQuestion pairs.

You receive a source decision (current meeting) and a list of target open questions (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source decision → target open question only.

Not resolves:
- The decision answers only a subcase or instance of the question, or narrows it without fully settling it.
  Counter-example: "Use Redis for session storage" is NOT resolves for "What caching strategy should we adopt across all services?" — the question is broader than the decision's scope → null.
- The decision is phased, provisional, or temporary and the question is broader in scope.
  Counter-example: "Build a phase-one automated ticket router" is NOT resolves for "Should we build or buy the long-term ticket routing capability?" — the decision settles the initial phase, not the long-term question → null.
- The decision assumes or presupposes an answer to the question rather than providing one.
  Counter-example: "Deploy the reporting warehouse on BigQuery" is NOT resolves for "What cloud provider will we use?" — the decision presupposes a provider but does not answer the provider question → null.

resolves — when to use it:
The decision gives a direct, complete, and final-enough answer to the open question so it should no longer be tracked as open. Partial answers, subcases, or scope-narrowing decisions do not qualify.
- Example: "Use blue/green deployments for all production releases" resolves "What deployment strategy should we use?".
- Example: "Store all EU tenant data in eu-central-1" resolves "Where should EU tenant data be hosted?".
- Example: "Adopt RabbitMQ as the asynchronous job queue" resolves "What queueing technology should background jobs use?".

Selection:
  1. Use resolves if the decision provides a direct, complete answer that closes the open question.
  2. Otherwise, use null.
"""

_DECISION_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → ActionItem pairs.

You receive a source decision (current meeting) and a list of target action items (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source decision → target action item only.

Not blocks:
- The decision changes the context or parameters of the action item but the action item can still proceed by incorporating that change.
  Counter-example: "Cap exchange-rate cache TTL to 2 min for settlement calculations" is NOT blocks for "Benchmark cache hit rates under peak load" — the decision changes a parameter but doesn't prevent the benchmarking → null.
- The decision makes the action item redundant or obsolete by replacing the system or tool the action item was targeting — "no longer needed" is not the same as "prevented from executing."
  Counter-example: "Standardise on Grafana for all dashboards — Datadog is out" is NOT blocks for "Configure Datadog alert thresholds for the payments service" — the action item is no longer needed, but the decision doesn't impose a constraint that prevents it from proceeding as stated → null.
- The decision shares the same domain or system but does not directly prevent execution.
  Counter-example: "Enable mutual TLS for all inter-service calls" is NOT blocks for "Set up CI/CD pipeline for the frontend" — same infrastructure area, independent tasks → null.

blocks — when to use it:
The decision imposes a restriction, freeze, prohibition, or incompatible constraint that prevents the action item from proceeding as stated. If the action item can proceed by incorporating the decision as a new constraint or updating its deliverable, use null.
- Example: "Defer all non-critical deployments pending security review" blocks "Deploy new billing service to production".
- Example: "Freeze all schema changes until migration is complete" blocks "Add nullable audit_ts column to events table".
- Example: "Require legal sign-off before any PII export" blocks "Export historical user activity logs for the data science team".

Selection:
  1. Ask: could someone execute this action item right now if they chose to? If yes but it would be pointless or wasteful, use null — blocks requires that execution is physically, legally, or logically impossible, not merely unnecessary.
  2. Use blocks if the decision imposes a restriction or freeze that makes execution impossible as stated.
  3. Otherwise, use null.
"""

_PROMPTS: dict[ConceptType, str] = {
    ConceptType.RISK: _DECISION_RISK_PROMPT,
    ConceptType.OPEN_QUESTION: _DECISION_OPEN_QUESTION_PROMPT,
    ConceptType.ACTION_ITEM: _DECISION_ACTION_ITEM_PROMPT,
}


_RESULT_MODELS = {
    ConceptType.RISK: _DecisionToRiskResult,
    ConceptType.OPEN_QUESTION: _DecisionToOpenQuestionResult,
    ConceptType.ACTION_ITEM: _DecisionToActionItemResult,
}


class DecisionCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent):
    """Resolves Decision → Risk (mitigates), Decision → OpenQuestion (resolves), Decision → ActionItem (blocks)."""

    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig, target_type: ConceptType) -> None:
        super().__init__(llm=llm, config=config)
        self._target_type = target_type

    @property
    def _result_model(self) -> type[_DecisionResult]:
        return _RESULT_MODELS[self._target_type]

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
