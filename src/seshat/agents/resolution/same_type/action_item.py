from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _EntryBase, _ResultBase
from seshat.models.enums import RelationshipType


class _ActionItemEntry(_EntryBase):
    rel_type: (  # type: ignore[override]
        Literal[
            RelationshipType.SUPERSEDES,
            RelationshipType.AMENDS,
            RelationshipType.CONFLICTS_WITH,
            RelationshipType.BLOCKS,
            RelationshipType.DEPENDS_ON,
        ]
        | None
    )


class _ActionItemResult(_ResultBase[_ActionItemEntry]): ...


_ACTION_ITEM_PROMPT = """\
You are an action item relationship resolution agent.

You receive a source action item (current meeting) and a list of target action items (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source → target only.

Not supersedes:
- The target task is still needed; the source only precedes or modifies it.
  Counter-example: "Write rollback plan" is NOT supersedes for "Deploy PgBouncer to production" — the deployment is still needed → blocks.
- When ambiguous between supersedes and amends, prefer amends.

Not conflicts_with:
- The source and target are sequentially ordered — one must complete before the other can start.
  Counter-example: "Publish the API documentation" is NOT conflicts_with "Finalise the API contract" — documentation cannot be published until the contract is finalised → depends_on.
- conflicts_with requires both tasks to be simultaneously assigned and mutually incompatible as stated — they cannot both be executed.

Not depends_on:
- The source provides input to the target, not the other way around.
  Counter-example: "Supply initial concurrency limits" is NOT depends_on "Drive rate limiting policy" — it is an input to it, not a prerequisite for it → null.
- The source and target are both preparation steps for the same initiative but can proceed in parallel — sharing a goal is not a dependency.
  Counter-example: "Draft the incident response playbook for the new auth service" is NOT depends_on "Set up alerting rules for the new auth service" — both prepare the auth service for production, but writing the playbook does not require alerting to be configured first → null.
- depends_on is anti-symmetric: if A depends_on B, then B does not depend_on A.

Not blocks:
- The source and target are both work items on the same component or initiative, but completing one does not gate the other.
  Counter-example: "Migrate the billing service to the new database schema" does NOT block "Update the billing service API documentation" — both are billing service tasks, but the documentation can be written before or after the migration → null.
- blocks is anti-symmetric: if A blocks B, then B does not block A.
- Assign it in the direction where completion of the source is genuinely required before the target can start.

supersedes — when to use it:
The source takes the place of the target — the target task is no longer needed or has been absorbed.
- Example: "Alice will rewrite the full migration script" supersedes "Alice will patch the migration script".

amends — when to use it:
The source modifies the target (qualifies, narrows, or extends) without replacing it.
- Example: "Alice will rewrite the migration script by Friday EOD" amends "Alice will rewrite the migration script".

conflicts_with — when to use it:
Both action items assign contradictory ownership or intent to the same task — both cannot be true simultaneously.
- Example: "Alice will own the migration rewrite" conflicts_with "Bob will own the migration rewrite".

blocks — when to use it:
The source task must complete before the target can proceed. The target is still needed but cannot start.
- Example: "Write rollback plan" blocks "Deploy PgBouncer to production".

depends_on — when to use it:
The source task requires the target to be completed to be actionable or coherent.
- Example: "Run integration tests" depends_on "Provision test environment".

Boundary — conflicts_with vs depends_on (minimal-diff pair):
- "Alice will own the on-call rotation redesign" → conflicts_with "Bob will own the on-call rotation redesign" (same task, contradictory ownership — both active, cannot both be true)
- "Publish the API documentation" → depends_on "Finalise the API contract" (sequential — documentation cannot be published until the contract is settled)

Selection:
  1. Use supersedes if the target is no longer needed or has been absorbed by the source.
  2. Use conflicts_with if both are active and assign contradictory ownership or intent to the same task.
  3. Use blocks if the source must complete before the target can start (target still needed).
  4. Use depends_on if the source cannot proceed without the target being completed first.
  5. Use amends if the source modifies the target without replacing it.
  6. Otherwise, use null.
"""


class ActionItemResolutionAgent(BaseSameTypeResolutionAgent[_ActionItemEntry]):
    @property
    def _result_model(self) -> type[_ActionItemResult]:
        return _ActionItemResult

    @property
    def _system_prompt(self) -> str:
        return _ACTION_ITEM_PROMPT
