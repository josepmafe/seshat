from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from seshat.agents.identification.reflective import NodeReview, SelfReviewResult
from seshat.agents.resolution.base import ResolvedRelationship
from seshat.agents.resolution.reflective import (
    ReflectiveResolutionAgent,
    _SelfReviewRetryExhaustedError,
)
from seshat.agents.resolution.same_type.decision import DecisionResolutionAgent
from seshat.config.settings import ResolutionLLMConfig
from seshat.models.enums import RelationshipType
from tests.helpers import make_node, make_structured_llm


def _make_inner() -> DecisionResolutionAgent:
    return DecisionResolutionAgent(llm=make_structured_llm(), config=ResolutionLLMConfig())


def _rel(src_id=None, tgt_id=None, rel_type=RelationshipType.SUPERSEDES) -> ResolvedRelationship:
    return ResolvedRelationship(
        source_id=src_id or uuid4(),
        target_id=tgt_id or uuid4(),
        rel_type=rel_type,
        rationale="test",
    )


def _all_pass(n: int) -> SelfReviewResult:
    return SelfReviewResult(reviews=[NodeReview(passed=True) for _ in range(n)])


def _mixed(passed: list[bool], rationales: list[str | None] | None = None) -> SelfReviewResult:
    rationales = rationales or [None] * len(passed)
    reviews = [NodeReview(passed=p, rationale=r) for p, r in zip(passed, rationales, strict=True)]
    return SelfReviewResult(reviews=reviews)


class TestReflectiveResolutionAgent:
    async def test_returns_all_relationships_when_validation_passes(self):
        src = make_node("src")
        tgt = make_node("tgt")
        inner = _make_inner()
        rels = [_rel(src.id, tgt.id)]
        inner._run_for_source = AsyncMock(return_value=rels)
        review_llm = make_structured_llm(return_value=_all_pass(1))

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result = await agent._run_for_source(source=src, targets=[tgt])

        assert result == rels

    async def test_filters_failed_relationships(self):
        src = make_node("src")
        tgt1 = make_node("tgt1")
        tgt2 = make_node("tgt2")
        inner = _make_inner()
        good = _rel(src.id, tgt1.id, RelationshipType.SUPERSEDES)
        bad = _rel(src.id, tgt2.id, RelationshipType.AMENDS)
        inner._run_for_source = AsyncMock(return_value=[good, bad])
        review_llm = make_structured_llm(
            return_value=_mixed([True, False], [None, "Wrong rel_type — AMENDS doesn't apply here."])
        )

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result = await agent._run_for_source(source=src, targets=[tgt1, tgt2])

        assert result == [good]

    async def test_returns_empty_when_inner_returns_empty(self):
        src = make_node("src")
        tgt = make_node("tgt")
        inner = _make_inner()
        inner._run_for_source = AsyncMock(return_value=[])
        review_llm = make_structured_llm()

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result = await agent._run_for_source(source=src, targets=[tgt])

        assert result == []
        review_llm.with_structured_output.assert_not_called()

    async def test_falls_back_to_all_relationships_on_validation_exhaustion(self):
        src = make_node("src")
        tgt = make_node("tgt")
        inner = _make_inner()
        rels = [_rel(src.id, tgt.id)]
        inner._run_for_source = AsyncMock(return_value=rels)
        inner._retryable_structured_ainvoke = AsyncMock(side_effect=_SelfReviewRetryExhaustedError("exhausted"))

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        result = await agent._run_for_source(source=src, targets=[tgt])

        assert result == rels

    async def test_falls_back_to_all_relationships_on_count_mismatch(self):
        src = make_node("src")
        tgt1 = make_node("tgt1")
        tgt2 = make_node("tgt2")
        inner = _make_inner()
        rels = [_rel(src.id, tgt1.id), _rel(src.id, tgt2.id)]
        inner._run_for_source = AsyncMock(return_value=rels)
        review_llm = make_structured_llm(return_value=_all_pass(1))

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result = await agent._run_for_source(source=src, targets=[tgt1, tgt2])

        assert result == rels

    def test_delegates_system_prompt_to_inner(self):
        inner = _make_inner()
        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        assert agent._system_prompt == inner._system_prompt

    def test_delegates_result_model_to_inner(self):
        inner = _make_inner()
        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        assert agent._result_model is inner._result_model

    def test_delegates_validate_relationships_to_inner(self):
        inner = _make_inner()
        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        src, tgt = uuid4(), uuid4()
        rels = [_rel(src, tgt, RelationshipType.SUPERSEDES), _rel(tgt, src, RelationshipType.SUPERSEDES)]
        valid, dropped = agent._validate_relationships(rels)
        assert valid == []
        assert len(dropped) == 2

    def test_prompt_texts_includes_validate_prompt(self):
        inner = _make_inner()
        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        texts = agent.prompt_texts()
        assert "validate" in texts
        assert "system" in texts
