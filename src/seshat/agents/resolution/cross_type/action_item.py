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
For each target, determine whether completing the action item would directly mitigate the risk.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source action item → target risk only.
- Require direct causal coupling — completing the action item must reduce the specific failure mode, likelihood, severity, exposure, or blast radius of the risk.
- An action item that only investigates, discusses, plans, proposes, documents, monitors, dashboards, alerts, or gathers input about a risk does NOT mitigate the underlying technical or operational failure mode.
- An action item that creates an implementation plan, proposal, design, roadmap, ticket breakdown, or recommendation does NOT mitigate the risk unless the same action item also deploys, enables, enforces, or completes the control.
- Investigation, planning, documentation, dashboards, monitoring, or alerts qualify as mitigates only when the target risk's stated failure mode is missing information, missing documentation, missing monitoring, missing alerting, or missing ownership.
- An action item that is only related to the same system, component, or project does NOT qualify.
- An action item that only indirectly reduces a downstream consequence of the risk does NOT qualify; it must address the risk's stated failure mode directly.
- Prefer null when the source merely provides context, motivation, evidence, constraints, or implementation detail for the target without changing whether the target is mitigated.

Selection priority:
  1. Use mitigates if completing the action item would directly reduce or control the risk's failure mode.
  2. Otherwise, use null.

Relationship definitions:
- mitigates: completing the source action item directly implements or operationalizes a concrete control that reduces the target risk's likelihood, severity, exposure, or blast radius.
    The mitigation must be an expected effect of completing the task itself, not merely a possible downstream benefit, future implementation step, proposal, plan, design, roadmap, ticket breakdown, or new information produced by validation.
    - Example: action_item "Add disk-usage alert for export workers" mitigates risk "Missing proactive alerting on export worker disk utilisation".
    - Example: action_item "Implement checkout request concurrency caps" mitigates risk "Checkout bursts may exhaust payment gateway capacity".
    - Example: action_item "Add cold-storage retrieval path for archived documents" mitigates risk "Short document lifecycle policies may break archived document retrieval workflows".
- null: the action item does not directly mitigate the risk, or is only exploratory/contextual.
    - Example: action_item "Discuss hosted search options with platform team" and risk "Search deployment model remains uncertain" — discussion may inform a later decision but does not itself reduce the uncertainty.
    - Example: action_item "Write incident summary" and risk "Billing workers will miss processing deadlines if invoice volume doubles" — documentation does not reduce the worker load.
    - Example: action_item "Sync on updated metric names" and risk "Payment gateway capacity will degrade under checkout scale-out" — same operational area, but no direct mitigation.
    - Example: action_item "Tighten billing worker lag alerts" and risk "Billing workers will miss processing deadlines if invoice volume doubles" — alerts detect or escalate the breach but do not reduce the load or processing cost that causes it.
    - Example: action_item "Convert billing optimisation proposal into an implementation plan" and risk "Billing workers will miss processing deadlines if invoice volume doubles" — an implementation plan may describe useful controls, but it does not mitigate the risk until those controls are deployed or enabled.
    - Example: action_item "Break search indexing optimisation into implementation tickets" and risk "Search indexing jobs may exceed the nightly processing window" — ticket breakdown prepares future mitigation work but does not itself reduce indexing time.
    - Example: action_item "Draft uploaded-document lifecycle policy proposal" and risk "No uploaded-document lifecycle policy creates cost and compliance exposure" — drafting a proposal starts governance work but does not mitigate the risk unless it produces an approved or enforceable policy.

Counter-examples:
- action_item "Evaluate shared Redis session compatibility" does NOT mitigate risk "Shared Redis sessions may break tenant isolation" unless the task includes implementing the compatibility fix — evaluation alone informs the risk.
- action_item "Clarify uploaded-document lifecycle requirements with legal" does NOT mitigate risk "No uploaded-document lifecycle policy creates cost and compliance exposure" unless it produces or enforces the policy — clarification alone is an input.
- action_item "Create automated ticket routing proposal" does NOT mitigate risk "Automated ticket routing alerts may become noise without operations triage" unless it assigns triage ownership or defines enforceable acceptance criteria.
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
