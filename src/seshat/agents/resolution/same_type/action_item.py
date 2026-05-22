from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import RelationshipType


class _ActionItemEntry(_EntryBase):
    rel_type: (
        Literal[
            RelationshipType.SUPERSEDES,
            RelationshipType.AMENDS,
            RelationshipType.CONFLICTS_WITH,
            RelationshipType.BLOCKS,
            RelationshipType.DEPENDS_ON,
        ]
        | None
    )  # type: ignore[override]


class _ActionItemResult(_ResultBase[_ActionItemEntry]): ...


_ACTION_ITEM_PROMPT = """\
You are an action item relationship resolution agent.

You receive a source action item (current meeting) and a list of target action items (prior meeting KB).
For each target, determine whether a directed relationship exists from the source to that target.

Rules:
- Every target MUST appear in the output — including those with rel_type=null.
- Relationships are directed: source → target only.
- Assign at most one non-null rel_type per (source, target) pair.
- Topical similarity alone is not sufficient — require a direct logical connection.
- When ambiguous between supersedes and amends, prefer amends.
- amends is directed from the more specific to the more general: the source refines or extends the target. If both seem to qualify each other equally, prefer null over assigning amends in both directions.
- blocks is anti-symmetric: if A blocks B, then B does not block A. Assign it in the direction where completion of the source is genuinely required before the target can start.
- depends_on is anti-symmetric: if A depends_on B, then B does not depend_on A. Assign it only in the direction where the source cannot proceed without the target.
- Do not use depends_on when the source provides input to the target — that is the target's dependency, not the source's.
    Counter-example: "Supply initial concurrency limits" does NOT depend_on "Drive rate limiting policy" — it is an input to it.
- Allowed relationships: supersedes, amends, conflicts_with, blocks, depends_on, null.
- Prohibited relationships: none.

Relationship definitions:
- supersedes: the source takes the place of the target — the target task is no longer needed or has been absorbed.
    Example: "Alice will rewrite the full migration script" supersedes "Alice will patch the migration script".
- amends: the source modifies the target (qualifies, narrows, or extends) without replacing it.
    Example: "Alice will rewrite the migration script by Friday EOD" amends "Alice will rewrite the migration script".
- conflicts_with: both action items assign contradictory ownership or intent to the same task — both cannot be true simultaneously.
    Example: "Alice will own the migration rewrite" conflicts_with "Bob will own the migration rewrite".
- blocks: the source task must complete before the target can proceed — the target is still needed but cannot start.
    Distinguish from supersedes: if the target is still needed, use blocks; if it is no longer needed, use supersedes.
    Example: "Write rollback plan" blocks "Deploy PgBouncer to production".
- depends_on: the source task requires the target to be completed to be actionable or coherent.
    Example: "Run integration tests" depends_on "Provision test environment".
- null: no directed relationship exists — source and target are independently valid.
"""


class ActionItemResolutionAgent(BaseSameTypeResolutionAgent[_ActionItemEntry]):
    @property
    def _result_model(self) -> type[_ActionItemResult]:
        return _ActionItemResult

    @property
    def _system_prompt(self) -> str:
        return _ACTION_ITEM_PROMPT
