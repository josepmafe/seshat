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
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source open question → target decision only.

Not blocks:
- The question is relevant to the decision's context but the decision can still be executed with reasonable defaults or documented assumptions.
  Counter-example: "What logging format should we standardise on?" is NOT blocks for "Deploy centralised log aggregation to production" — deployment can proceed regardless of format → null.
- The question and decision share a domain but the question does not logically precede the decision.
  Counter-example: "What SLA can we commit to for the reporting API?" is NOT blocks for "Set up read replicas for the analytics database" — the SLA question is downstream of infrastructure, not a prerequisite → null.
- The question concerns a different aspect of the same system without gating execution.
  Counter-example: "Should we use gRPC or REST for inter-service calls?" is NOT blocks for "Add circuit breakers to all service calls" — circuit breakers are protocol-agnostic → null.

blocks — when to use it:
The question leaves a required prerequisite, parameter, or constraint unresolved so the decision cannot be safely executed or treated as final.
- Example: "What cloud provider will we use?" blocks "Deploy the reporting warehouse on BigQuery" — the provider must be chosen before a provider-specific deployment is committed.
- Example: "Which compliance framework applies to EU tenant data?" blocks "Store EU tenant data in eu-central-1" — regulatory requirements must be known first.

Selection:
  1. Use blocks if the decision cannot be safely or consistently executed until the question is answered.
  2. Otherwise, use null.
"""

_OPEN_QUESTION_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating OpenQuestion → ActionItem pairs.

You receive a source open question (current meeting) and a list of target action items (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source open question → target action item only.

Not blocks:
- The question is relevant to the action item's context but the action item can begin with documented assumptions or proceed independently.
  Counter-example: "What monitoring stack should we adopt?" is NOT blocks for "Add structured logging to the billing service" — structured logging is useful regardless of which monitoring stack is chosen → null.
- The question and action item share a domain but the question does not logically precede the task.
  Counter-example: "What SLA can we commit to for the reporting API?" is NOT blocks for "Add indexes to the reports table" — the indexing task can proceed regardless of the SLA → null.
- The action item's purpose is to answer, investigate, or clarify the question — the question cannot block its own investigation.
  Counter-example: "What lifecycle exceptions are required for legal holds?" is NOT blocks for "Clarify uploaded-document lifecycle requirements with legal" — the action item is meant to answer the question → null.

blocks — when to use it:
The question withholds a required input, parameter, constraint, or direction so the action item cannot be meaningfully started or completed.
- Example: "What lifecycle policy applies to uploaded documents?" blocks "Configure object-storage lifecycle settings" — the settings are directly determined by the policy answer.
- Example: "Which cloud provider will we use?" blocks "Provision the production Kubernetes cluster" — provisioning is provider-specific.

Selection:
  1. Use blocks if the action item cannot be meaningfully started or completed until the question is answered.
  2. Otherwise, use null.
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
