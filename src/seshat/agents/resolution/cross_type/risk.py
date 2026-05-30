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
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source risk → target decision only.

Not blocks:
- The risk describes a concern, consequence, or tradeoff related to the decision without making it impossible or definitionally wrong to act on.
  Counter-example: "Third-party vendor may not meet SLA guarantees" is NOT blocks for "Use vendor X for log aggregation" — the risk is a consequence of the decision, not a concrete obstacle to adopting it → null.
- The risk motivates, informs, or is addressed by the decision without preventing execution.
  Counter-example: "Automated incident classification alerts may become noise without triage ownership" is NOT blocks for "Defer incident classification until triage ownership is clarified" — the risk motivates the deferral but does not prevent executing it → null.
- The risk is a symptom of the current situation, not a prerequisite the decision depends on.
  Counter-example: "Teams unfamiliar with the new deployment process may skip the approval gate" is NOT blocks for "All teams must use the standard deployment process" — the risk describes a hazard during rollout; the decision can be adopted regardless → null.

blocks — when to use it:
The risk, if unresolved, makes it impossible or definitionally wrong to act on the decision as stated. The decision cannot be safely enacted because a concrete prerequisite is missing — not merely risky or suboptimal.
- Example: "Legal sign-off pending on data residency" blocks "Deploy EU tenant data to region X" — the legal constraint must be known before the decision can be enacted.
- Example: "Certificate authority validation unavailable in eu-west-1" blocks "Enforce mutual TLS for all services in eu-west-1" — the decision literally cannot be executed.

Boundary — blocks vs null:
- "Load test shows storage layer cannot handle peak throughput" → blocks "Migrate all writes to the new storage layer" (executing the decision would concretely fail)
- "Storage layer may degrade under sustained write load" → null "Standardise on the new storage layer for non-critical services" (risk is a concern; the decision can still be adopted)

Selection:
  1. Use blocks if the risk makes it impossible or definitionally wrong to act on the decision as stated.
  2. Otherwise, use null.
"""

_RISK_OPEN_QUESTION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Risk → OpenQuestion pairs.

You receive a source risk (current meeting) and a list of target open questions (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source risk → target open question only.

Not blocks:
- The risk informs or motivates the question but a defensible answer can still be given despite the risk.
  Counter-example: "A managed search vendor may not meet latency targets" is NOT blocks for "Should we use hosted, self-managed, or database-native search?" — the risk informs one option but the question can still be analysed and answered → null.
- The risk is a symptom of the situation — the question can be answered regardless of whether the risk is resolved.
  Counter-example: "Engineers unfamiliar with the new auth service may escalate incidents incorrectly" is NOT blocks for "Should we consolidate on-call ownership across all services?" — the risk follows from expansion decisions; it doesn't gate the ownership question → null.

blocks — when to use it:
The risk makes every possible answer to the question premature or definitionally invalid until it is resolved. The risk withholds a constraint, fact, or regulatory outcome that determines what answers are even permissible — not just one factor to weigh.
- Example: "Legal review still pending on data residency" blocks "Where should EU tenant data be hosted?" — no valid answer exists until the legal constraint is known.
- Example: "Compliance audit may prohibit storing session tokens longer than 24 h" blocks "What session token TTL should we use?" — the TTL answer depends directly on the audit outcome.

Boundary — blocks vs null:
- "Load test shows payment gateway cannot sustain peak checkout traffic" → blocks "Should we route all checkouts through the new gateway this quarter?" (the data makes every answer to the question invalid until performance is resolved)
- "Third-party vendor may not meet SLA guarantees" → null "Which vendor should we use for log aggregation?" (risk informs the evaluation; a defensible answer can still be given)

Selection:
  1. Use blocks only if every possible answer to the question would be premature or definitionally invalid until the risk is resolved.
  2. Otherwise, use null.
"""

_RISK_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Risk → ActionItem pairs.

You receive a source risk (current meeting) and a list of target action items (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source risk → target action item only.

Not blocks:
- The risk raises concerns relevant to the action item's domain but the task can still be executed or started.
  Counter-example: "Search index may saturate under high read load" is NOT blocks for "Archive old audit logs to cold storage" — same storage area, but the archiving task is not obstructed → null.
- The action item's purpose is to investigate, validate, or mitigate the risk — the risk cannot block its own resolution work.
  Counter-example: "Cache TTL may corrupt settlement calculations" is NOT blocks for "Evaluate safe cache TTLs for exchange-rate data" — the risk motivates the evaluation, not blocks it → null.
- The risk makes the action item's outcome less certain or more complex but the task can still proceed.
  Counter-example: "Refresh tokens stored in localStorage are vulnerable to XSS" is NOT blocks for "Update the internal SDK to support the new auth mechanism" — the storage risk is a concern in the same security domain, but SDK work can proceed independently → null.

blocks — when to use it:
The risk makes the action item impossible to complete correctly as stated — proceeding would produce incorrect or harmful results, or the task literally cannot be executed until the risk is resolved.
- Example: "Shared Redis session mode may break tenant isolation" blocks "Roll out shared Redis sessions to all web applications" — executing would concretely break isolation guarantees.
- Example: "Certificate authority for eu-west-2 is unavailable" blocks "Enable mutual TLS for all services in eu-west-2" — the task cannot be completed without a working CA.
- Example: "Load test shows current throughput limit is 200 RPS, not 500" blocks "Set checkout rate limiting threshold to 500 RPS in production" — the action would be based on incorrect data.

Selection:
  1. Use blocks if the action item cannot be completed correctly until the risk is resolved.
  2. Otherwise, use null.
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
