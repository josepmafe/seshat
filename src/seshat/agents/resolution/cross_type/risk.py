from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import ResolutionLLMConfig


class _CrossTypeRiskEntry(_EntryBase):
    rel_type: Literal[RelationshipType.BLOCKS] | None  # type: ignore[override]


class _CrossTypeRiskResult(_ResultBase[_CrossTypeRiskEntry]): ...


_RISK_DECISION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Risk → Decision pairs.

You receive a source risk (current meeting) and a list of target decisions (prior meeting KB).
For each target, determine whether the risk directly hinders or prevents the decision from being executed or enforced.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source risk → target decision only.
- Require a direct execution dependency — the risk must make the specific decision impossible or dangerous to execute as stated.
- A risk that raises concerns about a decision's consequences without stopping it does NOT block it.
- A risk that affects the same domain as the decision without directly obstructing its execution does NOT block it.
- A risk does NOT block a decision whose purpose is to defer, reject, investigate, or postpone the risky work unless the risk prevents that deferral or investigation itself.
- A risk that can be handled as an implementation constraint does NOT block the decision unless the decision cannot be safely acted on as stated.

Selection priority:
  1. Use blocks if the risk, if unresolved, would make it unsafe, illegal, or operationally impossible to act on the decision as stated.
  2. Otherwise, use null.

Relationship definitions:
- blocks: the source risk makes the target decision unsafe, illegal, operationally impossible, or invalid to execute/enforce as stated.
    The risk must be a concrete prerequisite or obstacle. A risk that merely describes a possible downside, tradeoff, or consequence of the decision is null.
    - Example: risk "Legal sign-off pending on data residency" blocks decision "Deploy EU tenant data to region X".
    - Example: risk "Load test results show the new storage layer cannot handle peak throughput" blocks decision "Migrate all writes to the new storage layer by end of quarter".
    - Example: risk "Certificate authority validation unavailable in the eu-west-1 region" blocks decision "Enforce mutual TLS for all services deployed in eu-west-1".
- null: the risk does not block the decision, or only raises concerns without stopping execution.
    - Example: risk "Stale exchange-rate cache corrupting settlement calculations" and decision "Adopt RabbitMQ as the asynchronous job queue" — they share a data-quality concern but the queueing decision can be written and adopted regardless.
    - Example: risk "Invoice export service degrades above 5 000 writes/min" and decision "Enforce code review for all production changes" — different domains entirely.
    - Example: risk "Automated incident classification alerts may become noise without operations triage" and decision "Defer incident classification until triage ownership is clarified" — the risk motivates the deferral but does not block executing the deferral decision.

Counter-examples:
- risk "Interim request caps becoming permanent technical debt" does NOT block decision "Set request limits for the billing API" — the risk flags a long-term concern, not an obstacle to the decision itself.
- risk "Third-party vendor may not meet SLA guarantees" does NOT block decision "Use vendor X for log aggregation" — the risk is a consequence of the decision, not a blocker; the decision can still be executed.
- risk "Document conversion may fail for files above 512 MB" does NOT block decision "Standardise on PDF/A for archived documents" — they share the document-processing domain but the format decision can be adopted regardless.
"""

_RISK_OPEN_QUESTION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Risk → OpenQuestion pairs.

You receive a source risk (current meeting) and a list of target open questions (prior meeting KB).
For each target, determine whether the risk directly prevents the open question from being meaningfully answered.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source risk → target open question only.
- Require a direct dependency — the risk must make it impossible or pointless to answer the question until the risk is resolved.
- A risk that is relevant to the question's answer without preventing the analysis does NOT block it.
- A risk in the same domain that does not constrain the question's answer space does NOT block it.
- A risk concerning one possible option does NOT block an open question about choosing among options unless the risk must be resolved before any option can be evaluated.
- A risk that should be considered as an input to the answer is null unless it makes every answer premature, unsafe, or operationally invalid.

Selection priority:
  1. Use blocks if, while the risk is unresolved, any answer to the open question would be premature, unsafe, or operationally invalid.
  2. Otherwise, use null.

Relationship definitions:
- blocks: the source risk makes the target open question unanswerable, premature, or operationally invalid until the risk is resolved.
    The risk must determine whether the question has a valid answer at all, not merely be one factor to consider while answering it.
    - Example: risk "Legal review still pending on data residency" blocks open_question "Where should EU tenant data be hosted?" — the legal constraint must be known before the question can be answered.
    - Example: risk "Load test shows the current payment gateway cannot sustain peak checkout traffic" blocks open_question "Should we route all checkouts through the new payment gateway this quarter?" — the performance data directly gates the decision.
    - Example: risk "Compliance audit may prohibit storing session tokens longer than 24 h" blocks open_question "What session token TTL should we use?" — the TTL answer depends on the audit outcome.
- null: the risk does not prevent the question from being analysed or answered, or only raises concerns in the same domain.
    - Example: risk "Design language inadvertently locks in a single search vendor" and open_question "Automated Ticket Routing: Build vs. Buy Decision" — both are platform choices but the build/buy analysis can proceed independently.
    - Example: risk "Pipeline may fail for messages above 512 KB" and open_question "What branching strategy should we use?" — unrelated concerns.
    - Example: risk "A managed search vendor may not meet latency targets" and open_question "Should we use hosted search, self-managed search, or database-native search?" — the risk informs option evaluation but does not prevent comparing the options.

Counter-examples:
- risk "Cache warming failure causes elevated latency after deployments" does NOT block open_question "What cache TTL should we use for product catalogue data?" — the risk is a consequence of a cache decision, not a prerequisite for making it; the question can still be analysed.
- risk "Stale exchange-rate cache may corrupt settlement calculations" does NOT block open_question "What monitoring stack should we adopt?" — same data concern but the monitoring question can be answered regardless.
- risk "Third-party vendor may not meet SLA guarantees" does NOT block open_question "Which vendor should we use for log aggregation?" — the risk informs the answer but does not make it impossible to evaluate options and decide.
"""

_RISK_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Risk → ActionItem pairs.

You receive a source risk (current meeting) and a list of target action items (prior meeting KB).
For each target, determine whether the risk directly prevents the action item from being safely or meaningfully executed.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source risk → target action item only.
- Require a direct execution dependency — the risk must make it unsafe, impossible, or operationally invalid to execute the specific action item.
- A risk that raises concerns about consequences without stopping execution does NOT block it.
- A risk that affects the same system or domain without directly obstructing the task does NOT block it.
- A risk does NOT block an action item whose purpose is to investigate, validate, document, or mitigate that risk, unless the action itself would be unsafe or impossible.
- A risk that changes how the action item should be done is null if the action can still proceed with that constraint.

Selection priority:
  1. Use blocks if proceeding with the action item while the risk is unresolved would be unsafe, illegal, or would cause the action item to fail or be meaningless.
  2. Otherwise, use null.

Relationship definitions:
- blocks: the source risk makes the target action item unsafe, impossible, operationally invalid, or meaningless to execute until the risk is resolved.
    The risk must create a specific, concrete obstacle to executing the task. If the task is investigation, validation, documentation, or mitigation work that can proceed because of the risk, use null.
    - Example: risk "Shared Redis session mode may break tenant isolation" blocks action_item "Roll out shared Redis sessions to all web applications" — rolling out would break isolation guarantees.
    - Example: risk "Certificate authority for eu-west-2 is unavailable" blocks action_item "Enable mutual TLS for all services in eu-west-2" — the task cannot be completed without a working CA.
    - Example: risk "Load test shows current throughput limit is 200 RPS, not 500" blocks action_item "Set checkout rate limiting threshold to 500 RPS in production" — the action would be based on incorrect capacity data.
- null: the risk does not prevent the action item from proceeding, or raises concerns without creating a concrete obstacle.
    - Example: risk "Interim request caps becoming permanent technical debt" and action_item "Billing Worker Scaling Proposal" — the risk flags a debt concern but the proposal work can proceed.
    - Example: risk "Invoice export service degrades above 5 000 writes/min" and action_item "Write runbook for on-call incident response" — same operations domain, no execution dependency.
    - Example: risk "Cache TTL may corrupt settlement calculations" and action_item "Evaluate safe cache TTLs for exchange-rate data" — the risk motivates the evaluation but does not block doing it.

Counter-examples:
- risk "Cache warming failure causes elevated latency after deployments" does NOT block action_item "Set up CI/CD pipeline for the admin frontend" — the risk concerns backend cache, not frontend CI/CD.
- risk "API rate limiting may allow a single client to exhaust capacity" does NOT block action_item "Implement OAuth provider integration" — they share the API layer but the OAuth implementation is not gated on the rate limiting concern.
- risk "Search index may saturate under high read load" does NOT block action_item "Archive old audit logs to cold storage" — the archiving task may interact with storage but is not obstructed by search read saturation.
"""

_PROMPTS: dict[ConceptType, str] = {
    ConceptType.DECISION: _RISK_DECISION_PROMPT,
    ConceptType.OPEN_QUESTION: _RISK_OPEN_QUESTION_PROMPT,
    ConceptType.ACTION_ITEM: _RISK_ACTION_ITEM_PROMPT,
}


class RiskCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent[_CrossTypeRiskEntry]):
    """Resolves Risk → Decision (blocks), Risk → OpenQuestion (blocks), Risk → ActionItem (blocks)."""

    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig, target_type: ConceptType) -> None:
        super().__init__(llm=llm, config=config)
        self._target_type = target_type

    @property
    def _result_model(self) -> type[_CrossTypeRiskResult]:
        return _CrossTypeRiskResult

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
