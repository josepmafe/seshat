from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import RelationshipType


class _RiskEntry(_EntryBase):
    rel_type: Literal[RelationshipType.AMENDS] | None  # type: ignore[override]


class _RiskResult(_ResultBase[_RiskEntry]): ...


_RISK_PROMPT = """\
You are a risk relationship resolution agent.

You receive a source risk (current meeting) and a list of target risks (prior meeting KB).
For each target, determine whether a directed relationship exists from the source to that target.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source → target only.
- Assign at most one non-null rel_type per (source, target) pair.
- Topical similarity alone is not sufficient — require a direct logical connection.
- amends is directed from the more specific to the more general: the source adds precision to the target. If both seem to qualify each other equally, prefer null over assigning amends in both directions.
- Allowed relationships: amends, null.
- Prohibited relationships: supersedes, conflicts_with, blocks, depends_on.

Selection priority:
  1. Use amends if the source narrows, scopes, quantifies, corrects, or adds concrete conditions to the target while preserving the same underlying risk concern.
  2. Otherwise, use null.

Relationship definitions:
- amends: the source refines the target risk without replacing it as a tracked concern.
    The source may make the target more specific, add a condition, identify a clearer trigger, quantify severity or likelihood, narrow the affected scope, or correct part of the risk description while preserving the same underlying concern.
    Use only when the source and target describe the same risk mechanism, failure mode, or concern at different levels of precision.
    - Example: "Pipeline may fail for messages above 512 KB" amends "Pipeline may fail for large messages".
    - Example: "Schema registry degrades above 5 000 writes/min" amends "Schema registry may become a bottleneck under load".
    - Example: "Cache warming failure causes elevated latency only on the first request after a deployment" amends "Cache warming failure may cause elevated latency".
- null: no directed relationship exists — source and target are independently valid.
    - Example: "Ingestion pipeline may exhaust memory on large payloads" and "Monitoring agent may introduce latency on metric collection" — different failure domains, no logical connection.
    - Example: "Database connection pool may be exhausted under sustained high read load" and "Database write-ahead log may fill up during a bulk import" — same component, unrelated failure modes.

Counter-examples:
- "Ingestion pipeline may fail for messages above 512 KB" does NOT have a null relationship with "Ingestion pipeline may fail for large messages" — the source adds precision to the target → amends.
- "API gateway becomes unavailable when upstream latency exceeds 30 s" does NOT have a null relationship with "API gateway may become unavailable under adverse conditions" — the source characterises the same concern more precisely → amends.
- "Database connection pool may be exhausted under sustained high read load" does NOT amend "Database write-ahead log may fill up during a bulk import" — same component, different failure mode → null.
"""


class RiskResolutionAgent(BaseSameTypeResolutionAgent[_RiskEntry]):
    @property
    def _result_model(self) -> type[_RiskResult]:
        return _RiskResult

    @property
    def _system_prompt(self) -> str:
        return _RISK_PROMPT
