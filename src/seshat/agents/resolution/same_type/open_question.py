from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import RelationshipType


class _OpenQuestionEntry(_EntryBase):
    rel_type: Literal[RelationshipType.AMENDS, RelationshipType.DEPENDS_ON] | None  # type: ignore[override]


class _OpenQuestionResult(_ResultBase[_OpenQuestionEntry]): ...


_OPEN_QUESTION_PROMPT = """\
You are an open question relationship resolution agent.

You receive a source open question (current meeting) and a list of target open questions (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source → target only.

Not amends:
- Source is broader than the target, not narrower.
  Counter-example: "What is our overall data residency policy?" is NOT amends for "Where should EU tenant data be stored?" — source is the general question, target is specific → null.
- Source and target are parallel questions at the same level of specificity; neither narrows the other.
  If both qualify each other equally, prefer null over assigning amends in either direction.

Not depends_on:
- The source can be meaningfully answered without settling the target first.
  Counter-example: "What logging format should we use?" is NOT depends_on "What monitoring stack should we adopt?" — both are answerable independently → null.
- depends_on is anti-symmetric: if A depends_on B, then B does not depend_on A.

amends — when to use it:
The source is a narrower subquestion, scoped variant, or concrete case of the target. The target remains open; the source adds a specific constraint, condition, or scope.
- Example: "What cache TTL is safe for compliance fields?" amends "What cache TTL should we use for profile data?"
- Example: "What retry policy should we use for failed webhook deliveries to EU customers?" amends "What retry policy should we use for failed webhook deliveries?"

depends_on — when to use it:
The source cannot be meaningfully answered without the target being settled first.
- Example: "What Kafka deployment model should we use?" depends_on "What cloud provider will we use?"
- Example: "What SLA can we commit to for the reporting API?" depends_on "What is the maximum acceptable query latency for the data warehouse?"

Boundary — amends vs null (minimal-diff pair):
- "What cache TTL should we use for compliance fields?" → amends "What cache TTL should we use for profile data?" (same question, source adds a scope constraint)
- "What cache TTL should we use for compliance fields?" → null "What retry budget should we apply to failed cache reads?" (same cache domain, different concern)

Selection:
  1. Use depends_on if the source cannot be meaningfully answered without the target being settled first.
  2. Use amends if the source is a narrower subquestion or scoped variant of the target.
  3. Otherwise, use null.
"""


class OpenQuestionResolutionAgent(BaseSameTypeResolutionAgent[_OpenQuestionEntry]):
    @property
    def _result_model(self) -> type[_OpenQuestionResult]:
        return _OpenQuestionResult

    @property
    def _system_prompt(self) -> str:
        return _OPEN_QUESTION_PROMPT
