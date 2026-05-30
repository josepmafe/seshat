from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import ResolutionLLMConfig


class _CrossTypeActionItemEntry(_EntryBase):
    rel_type: Literal[RelationshipType.MITIGATES] | None  # type: ignore[override]


class _CrossTypeActionItemResult(_ResultBase[_CrossTypeActionItemEntry]): ...


_ACTION_ITEM_RISK_PROMPT = """\
You are a cross-type relationship resolution agent evaluating ActionItem → Risk pairs.

You receive a source action item (current meeting) and a list of target risks (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source action item → target risk only.

Not mitigates:
- The action item investigates, discusses, plans, proposes, documents, monitors, or alerts about the risk without deploying, enabling, or enforcing a concrete control.
  Counter-example: "Evaluate shared Redis session compatibility" is NOT mitigates for "Shared Redis sessions may break tenant isolation" — evaluation informs the risk but does not reduce it → null.
- The action item creates a plan, proposal, design, or ticket breakdown — it does not mitigate until the controls described are actually deployed or enforced.
  Counter-example: "Convert billing optimisation proposal into an implementation plan" is NOT mitigates for "Billing workers will miss processing deadlines if invoice volume doubles" → null.
- The action item is related to the same system or project but does not address the risk's specific failure mode.
  Counter-example: "Sync on updated metric names" is NOT mitigates for "Payment gateway capacity will degrade under checkout scale-out" — same operational area, different mechanisms → null.

mitigates — when to use it:
Completing the action item directly implements or operationalises a concrete control that reduces the risk's likelihood, severity, exposure, or blast radius. The mitigation must be an expected effect of completing the task itself.
- Example: "Implement checkout request concurrency caps" mitigates "Checkout bursts may exhaust payment gateway capacity".
- Example: "Add disk-usage alert for export workers" mitigates "Missing proactive alerting on export worker disk utilisation".
- Example: "Add cold-storage retrieval path for archived documents" mitigates "Short document lifecycle policies may break archived document retrieval workflows".

Selection:
  1. Use mitigates if completing the action item directly implements a control that reduces the risk's failure mode.
  2. Otherwise, use null.
"""


_PROMPTS: dict[ConceptType, str] = {
    ConceptType.RISK: _ACTION_ITEM_RISK_PROMPT,
}


class ActionItemCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent[_CrossTypeActionItemEntry]):
    """Resolves ActionItem → Risk (mitigates)."""

    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig, target_type: ConceptType):
        super().__init__(llm=llm, config=config)
        self._target_type = target_type

    @property
    def _result_model(self) -> type[_CrossTypeActionItemResult]:
        return _CrossTypeActionItemResult

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
