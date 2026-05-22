from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import RelationshipType


class _OpenQuestionEntry(_EntryBase):
    rel_type: Literal[RelationshipType.AMENDS, RelationshipType.DEPENDS_ON] | None  # type: ignore[override]


class _OpenQuestionResult(_ResultBase[_OpenQuestionEntry]): ...


_OPEN_QUESTION_PROMPT = """\
You are an open question relationship resolution agent.

You receive a source open question (current meeting) and a list of target open questions (prior meeting KB).
For each target, determine whether a directed relationship exists from the source to that target.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source → target only.
- Assign at most one non-null rel_type per (source, target) pair.
- Topical similarity alone is not sufficient — require a direct logical connection.
- amends is directed from the more specific question to the more general question: the source is a narrower subquestion, scoped variant, or concrete case of the target. If both seem to qualify each other equally, prefer null over assigning amends in both directions.
- depends_on is anti-symmetric: if A depends_on B, then B does not depend_on A. Assign it only in the direction where the source cannot be meaningfully answered without the target being settled first.
- Allowed relationships: amends, depends_on, null.
- Prohibited relationships: supersedes, conflicts_with, blocks.

Selection priority:
  1. Use depends_on if the target must be answered first before the source can be meaningfully answered.
  2. Otherwise, use amends if the source narrows, specialises, scopes, or adds a concrete subquestion to the target while the target remains open.
  3. Otherwise, use null.

Relationship definitions:
- amends: the source question narrows, specialises, or adds a concrete constraint to the target question without replacing it.
    The target question remains valid and open, but the source identifies a more specific aspect, scope, condition, or case that should be considered when answering it.
    Use only when answering the source would contribute directly to answering the target, but would not fully answer or close the target.
    - Example: "What is the safe cache TTL for compliance fields?" amends "What cache TTL should we use for profile data?".
    - Example: "What retry policy should we use for failed webhook deliveries to EU customers?" amends "What retry policy should we use for failed webhook deliveries?".
    - Example: "Which data residency rule applies to German healthcare customers?" amends "What data residency rules apply to EU customers?".
- depends_on: the source question requires the target to be answered first to be answerable or coherent.
    - Example: "What Kafka deployment model should we use?" depends_on "What cloud provider will we use?".
    - Example: "What SLA can we commit to for the reporting API?" depends_on "What is the maximum acceptable query latency for the data warehouse?".
- null: no directed relationship exists — source and target are independently valid.
    - Example: "What branching strategy should we use for hotfixes?" and "What is the budget approval process for third-party tools?" — different concerns, no logical connection.
    - Example: "What logging format should we standardize on?" and "What monitoring stack should we adopt?" — same observability domain, but neither needs to be answered before the other.

Counter-examples:
- "What is the overall data residency policy?" does NOT amend "Where should EU tenant data be stored?" — the source is broader than the target, not narrower → null.
- "What cache TTL should we use for compliance fields?" does NOT have a null relationship with "What cache TTL should we use for profile data?" — the source narrows the target → amends.
- "What logging format should we standardize on?" does NOT depends_on "What monitoring stack should we adopt?" — both can be answered independently → null.
"""


class OpenQuestionResolutionAgent(BaseSameTypeResolutionAgent[_OpenQuestionEntry]):
    @property
    def _result_model(self) -> type[_OpenQuestionResult]:
        return _OpenQuestionResult

    @property
    def _system_prompt(self) -> str:
        return _OPEN_QUESTION_PROMPT
