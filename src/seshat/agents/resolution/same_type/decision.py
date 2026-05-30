from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import RelationshipType


class _DecisionEntry(_EntryBase):
    rel_type: Literal[RelationshipType.SUPERSEDES, RelationshipType.AMENDS, RelationshipType.CONFLICTS_WITH] | None  # type: ignore[override]


class _DecisionResult(_ResultBase[_DecisionEntry]): ...


_DECISION_PROMPT = """\
You are a decision relationship resolution agent.

You receive a source decision (current meeting) and a list of target decisions (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source → target only.

Not conflicts_with:
- The target is inactive (deferred, rejected, or on hold) — conflicts_with requires both decisions to be currently active.
- The source and target address different concern domains or architectural layers.
  Counter-example: "Require TLS for token endpoints" is NOT conflicts_with "Use session tokens for authentication" — transport security vs auth mechanism → null.

Not supersedes:
- The source is a temporary restriction (freeze, hold, moratorium) — it does not permanently replace the policy it restricts → null.
- The source narrows, qualifies, or adds an exception to the target without replacing it → amends instead.
- The source and target address different concern domains — decisions at different layers (auth mechanism vs token lifetime vs transport protocol) are null unless one explicitly governs the other.
- When ambiguous between supersedes and amends, prefer amends.

Not amends:
- The source and target are in different concern domains.
  Counter-example: "Require TLS for token endpoints" is NOT amends for "Use session tokens for authentication" — different concern domains → null.
- amends is directed from the more specific to the more general. If both qualify each other equally, prefer null.

conflicts_with — when to use it:
Both decisions are currently active, address the same concern, and are mutually incompatible — following one makes it impossible to follow the other.
- Example: "Set token lifetime to 15 minutes" conflicts_with "Set token lifetime to 24 hours".
- Example: "All services retry at most 3 times" conflicts_with "All services retry at most 10 times".

supersedes — when to use it:
The source permanently replaces the target in the same concern domain, rendering it no longer active.
- Example: "Use PostgreSQL for all storage" supersedes "Use SQLite for all storage".

amends — when to use it:
The source modifies the target (qualifies, narrows, extends, or adds an exception) without replacing it. Both address the same concern; the target remains broadly active.
- Example: "Enforce two-approver sign-off for production deployments only" amends "All deployments require two approvals".

Boundary — conflicts_with vs supersedes (minimal-diff pair):
- "All services retry at most 3 times" → conflicts_with "All services retry at most 10 times" (both active blanket policies, contradictory values — neither is yet inactive)
- "All services retry at most 3 times" → supersedes "Services may retry indefinitely" (source permanently closes out the old open-ended approach)

Selection:
  1. Use conflicts_with if both decisions are currently active, same concern, and mutually incompatible as stated. Skip if the target is inactive.
  2. Use supersedes if the source permanently replaces the target in the same concern domain.
  3. Use amends if the source qualifies, narrows, or extends the target in the same concern domain and the target remains broadly active.
  4. Otherwise, use null.
"""


class DecisionResolutionAgent(BaseSameTypeResolutionAgent[_DecisionEntry]):
    @property
    def _result_model(self) -> type[_DecisionResult]:
        return _DecisionResult

    @property
    def _system_prompt(self) -> str:
        return _DECISION_PROMPT
