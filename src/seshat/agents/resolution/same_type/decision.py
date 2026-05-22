from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import RelationshipType


class _DecisionEntry(_EntryBase):
    rel_type: Literal[RelationshipType.SUPERSEDES, RelationshipType.AMENDS, RelationshipType.CONFLICTS_WITH] | None  # type: ignore[override]


class _DecisionResult(_ResultBase[_DecisionEntry]): ...


_DECISION_PROMPT = """\
You are a decision relationship resolution agent.

You receive a source decision (current meeting) and a list of target decisions (prior meeting KB).
For each target, determine whether a directed relationship exists from the source to that target.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source → target only.
- Assign at most one non-null rel_type per (source, target) pair.
- Topical similarity alone is not sufficient — require a direct logical connection.
- A target decision is inactive if its title or description indicates it was deferred, rejected, or explicitly put on hold. Never assign conflicts_with to an inactive target.
- When ambiguous between supersedes and amends, prefer amends.
- amends is directed from the more specific to the more general: the source refines, narrows, or extends the target. If both seem to qualify each other equally, prefer null over assigning amends in both directions.
- Allowed relationships: supersedes, amends, conflicts_with, null.
- Prohibited relationships: blocks, depends_on.

Selection priority:
  1. Use supersedes if the source explicitly or functionally replaces the target, so the target should no longer be followed.
  2. Otherwise, use amends if the source can be read as a scoped exception, qualification, narrowing, extension, or parameter change to the target, and the target remains broadly active.
  3. Otherwise, use conflicts_with if the source and target are BOTH CURRENTLY ACTIVE decisions that cannot both be followed as stated. Skip this step if the target is deferred, rejected, or superseded.
  4. Otherwise, use null.

Relationship definitions:
- supersedes: the source takes the place of the target, rendering it no longer active.
    Use only when the target would no longer be followed.
    - Example: "Use PostgreSQL for all storage" supersedes "Use SQLite for all storage".
    - Example: "Deploy all services as Docker containers" supersedes "Deploy backend services as bare-metal processes".
- amends: the source modifies the target (qualifies, narrows, or extends) without replacing it.
    The target remains broadly active but with the source's qualification applied.
    - Example: "Enforce two-approver sign-off for production deployments only" amends "All deployments require two approvals".
    - Example: "Cap profile cache TTL to 2 min for compliance fields" amends "Use a 5-min profile cache TTL".
- conflicts_with: both decisions are currently active but mutually incompatible — both would need to be followed simultaneously but cannot both be true.
    A deferred, rejected, or superseded decision is not active — do not use conflicts_with when the target is inactive.
    A target is inactive if its title or description signals deferral ("deferred", "on hold", "rejected", "pending") or if the source supersedes it.
    To distinguish from amends: if you can append the source as an exception clause to the target without contradiction ("…except for X"), use amends. If following one makes it impossible to follow the other, use conflicts_with.
    - Example: "Use REST for all inter-service calls" conflicts_with "Use gRPC for all inter-service calls".
- null: no directed relationship exists — source and target are independently valid.
    - Example: "Use blue-green deployments for all releases" and "Store audit logs in a separate database" — different concerns, no logical connection.
    - Example: "All API responses must be paginated" and "All API responses must include a request ID header" — same concern area, but neither depends on or affects the other.

Counter-examples:
- "Require HTTPS on external-facing APIs only" does NOT supersede "All services must use HTTPS" — the general rule still applies to internal services; the source narrows it → amends.
- "Allow HTTP on internal health-check endpoints" does NOT conflicts_with "All services must use HTTPS" — the source qualifies the rule rather than contradicting it → amends.
- "Disable TLS certificate validation in staging" does NOT have a null relationship with "All environments must use valid TLS certificates" — both are active and directly incompatible → conflicts_with.
- "Adopt Redis for session storage" does NOT conflicts_with "Session storage technology selection — Deferred pending security review" — the target is inactive (deferred); use amends (narrowing the deferral scope) or supersedes (overriding it) instead.
"""


class DecisionResolutionAgent(BaseSameTypeResolutionAgent[_DecisionEntry]):
    @property
    def _result_model(self) -> type[_DecisionResult]:
        return _DecisionResult

    @property
    def _system_prompt(self) -> str:
        return _DECISION_PROMPT
