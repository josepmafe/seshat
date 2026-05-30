from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import RelationshipType


class _RiskEntry(_EntryBase):
    rel_type: Literal[RelationshipType.AMENDS] | None  # type: ignore[override]


class _RiskResult(_ResultBase[_RiskEntry]): ...


_RISK_PROMPT = """\
You are a risk relationship resolution agent.

You receive a source risk (current meeting) and a list of target risks (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source → target only.

Not amends:
- Source and target share the same component or domain but describe different failure modes.
  Counter-example: "Connection pool exhausted under high read load" is NOT amends for "Write-ahead log fills during bulk import" — same database, different mechanisms → null.
- Source is a parallel concern at the same level of precision, not a refinement of the target.
  Counter-example: "Cache stampede on cold start may spike database load" is NOT amends for "Connection pool exhausted under peak traffic" — related domain, independent failure modes → null.
- Both describe the same concern at the same specificity level; neither refines the other.
  If it is unclear which is more specific, prefer null over assigning amends in either direction.

amends — when to use it:
The source refines the target by adding precision to the same underlying failure mode. The target remains a valid tracked concern; the source makes it more specific. The source may:
- Narrow the trigger condition: "above 512 KB" amends "for large messages"
- Identify a more specific failure scenario: "during peak traffic" amends "under load"
- Correct or update the framing while preserving the same concern: "token exhaustion is most likely at peak traffic, not at cutover" amends "token exhaustion may occur at cutover"
- Quantify or add a concrete condition to an abstract statement

Boundary — amends vs null (minimal-diff pairs):
- "Pipeline may fail for messages above 512 KB" → amends "Pipeline may fail for large messages" (same failure mode, source adds a threshold)
- "Pipeline may fail for messages above 512 KB" → null "Pipeline may exhaust memory on very large file uploads" (different failure modes: message routing vs memory)

Selection:
  1. Use amends if source and target describe the same failure mode or concern, and the source refines, corrects, or adds precision to the target.
  2. Otherwise, use null.

"""


class RiskResolutionAgent(BaseSameTypeResolutionAgent[_RiskEntry]):
    @property
    def _result_model(self) -> type[_RiskResult]:
        return _RiskResult

    @property
    def _system_prompt(self) -> str:
        return _RISK_PROMPT
