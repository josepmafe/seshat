from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generic, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from seshat.agents.base import RetryExhaustedError
from seshat.agents.resolution.base import (
    ResolvedRelationship,
    _BaseResolutionAgent,
    _EntryBase,
    _ResultBase,
)
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel

    from seshat.models.nodes import KBNode

logger = get_logger(__name__)

E = TypeVar("E", bound=_EntryBase)


_VALIDATE_PROMPT = """\
Check each resolved relationship on two dimensions.

**Rule compliance** — check against the system prompt above:
- Is the assigned rel_type one of the allowed types for this (source, target) pair?
- Does the rationale logically support the chosen rel_type?

**Semantic compliance:**
- Does the source node's content actually justify the relationship to the target?
- Is the relationship direction correct (source → target, not the other way)?

Reject a relationship if its rel_type is wrong or if no relationship should exist between
these two nodes. Do not reject for minor phrasing or stylistic preference. When in doubt,
pass the relationship.

Return exactly one review per relationship, in the same order as the input list.

<resolved_relationships>\n{relationships_json}\n</resolved_relationships>
"""


class _SelfReviewRetryExhaustedError(RetryExhaustedError):
    pass


class RelationReview(BaseModel):
    passed: bool = Field(description="True if the item passes, False if it should be discarded.")
    rationale: str | None = Field(
        default=None,
        description="Brief explanation of which rule the item violates. Required when passed=False, else None.",
    )


class SelfReviewResult(BaseModel):
    reviews: list[RelationReview] = Field(
        description="One review per extracted item, in the same order as the input list.",
    )


class ReflectiveResolutionAgent(_BaseResolutionAgent[E], Generic[E]):
    """Wraps any _BaseResolutionAgent in a per-source validate→filter pass.

    After the inner agent resolves relationships for a source node, runs a single
    validation call that checks each relationship against the inner agent's own
    classification rules. Relationships that fail are discarded. Degrades gracefully
    to shallow behaviour if the validation call fails or returns a count mismatch.
    """

    def __init__(
        self,
        inner: _BaseResolutionAgent[E],
        review_llm: BaseChatModel,
    ) -> None:
        super().__init__(llm=inner._llm, config=inner._config)
        self._inner = inner
        self._review_llm = review_llm

    @property
    def _result_model(self) -> type[_ResultBase[E]]:
        return self._inner._result_model

    @property
    def _system_prompt(self) -> str:
        return self._inner._system_prompt

    def _validate_relationships(
        self,
        relationships: list[ResolvedRelationship],
    ) -> tuple[list[ResolvedRelationship], list[ResolvedRelationship]]:
        return self._inner._validate_relationships(relationships)

    def prompt_texts(self) -> dict[str, str]:
        return self._inner.prompt_texts() | {"validate": _VALIDATE_PROMPT}

    async def _run_for_source(
        self,
        source: KBNode,
        targets: list[KBNode],
        siblings: list[KBNode] | None = None,
    ) -> list[ResolvedRelationship]:
        relationships = await self._inner._run_for_source(source, targets, siblings)
        if not relationships:
            logger.debug("ReflectiveResolutionAgent: no relationships for source=%s", source)
            return []

        node_by_id: dict[UUID, KBNode] = {n.id: n for n in [source, *targets]}

        try:
            validation_result = await self._validate(relationships, node_by_id)
        except _SelfReviewRetryExhaustedError:
            logger.warning(
                "ReflectiveResolutionAgent: validation exhausted retries for source=%s — returning all relationships",
                source,
            )
            return relationships

        return self._filter(relationships, validation_result, source=source)

    async def _validate(
        self,
        relationships: list[ResolvedRelationship],
        node_by_id: dict[UUID, KBNode],
    ) -> SelfReviewResult:
        validation_data = [
            {
                "source": {"title": node_by_id[r.source_id].title, "description": node_by_id[r.source_id].description},
                "target": {"title": node_by_id[r.target_id].title, "description": node_by_id[r.target_id].description},
                "rel_type": r.rel_type.value,
                "rationale": r.rationale,
            }
            for r in relationships
            if r.source_id in node_by_id and r.target_id in node_by_id
        ]
        relationships_json = json.dumps(validation_data, indent=2)
        messages = [
            SystemMessage(
                content=[{"type": "text", "text": self._system_prompt, "cache_control": {"type": "ephemeral"}}]
            ),
            HumanMessage(content=_VALIDATE_PROMPT.format(relationships_json=relationships_json)),
        ]
        return await self._inner._retryable_structured_ainvoke(
            messages=messages,
            response_model=SelfReviewResult,
            raise_on_exhaustion=_SelfReviewRetryExhaustedError(
                f"ReflectiveResolutionAgent validate exhausted retries for source={next(iter(relationships)).source_id}"
            ),
            on_error_log_prefix="ReflectiveResolutionAgent.validate",
            llm=self._review_llm,
        )

    def _filter(
        self,
        relationships: list[ResolvedRelationship],
        validation: SelfReviewResult,
        source: KBNode,
    ) -> list[ResolvedRelationship]:
        if len(validation.reviews) != len(relationships):
            logger.warning(
                "ReflectiveResolutionAgent: count mismatch (%d reviews / %d relations) for source=%s. Returning all.",
                len(validation.reviews),
                len(relationships),
                source,
            )
            return relationships

        passing = [rel for rel, review in zip(relationships, validation.reviews, strict=True) if review.passed]
        failed_rationales = [
            f"({rel.source_id}→{rel.target_id} {rel.rel_type.value}): {review.rationale}"
            for rel, review in zip(relationships, validation.reviews, strict=True)
            if not review.passed
        ]

        if failed_rationales:
            logger.debug(
                "ReflectiveResolutionAgent: discarded %d/%d relationships for source=%s. Detail: %s",
                len(relationships) - len(passing),
                len(relationships),
                source,
                "; ".join(failed_rationales),
            )

        return passing
