import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.agents.resolution.cross_type.registry import CrossTypeResolutionRegistry
from seshat.agents.resolution.same_type.registry import SameTypeResolutionRegistry
from seshat.config.settings import ResolutionLLMConfig
from seshat.models.enums import ConceptType
from tests.helpers import make_node


def _make_same_type_registry() -> SameTypeResolutionRegistry:
    return SameTypeResolutionRegistry(llm=MagicMock(), config=ResolutionLLMConfig())


def _make_cross_type_registry() -> CrossTypeResolutionRegistry:
    return CrossTypeResolutionRegistry(llm=MagicMock(), config=ResolutionLLMConfig())


class TestSameTypeResolutionRegistry:
    def test_get_returns_agent_for_known_type(self):
        registry = _make_same_type_registry()
        for ct in (ConceptType.DECISION, ConceptType.RISK, ConceptType.ACTION_ITEM, ConceptType.OPEN_QUESTION):
            assert registry.get(ct) is not None

    def test_get_raises_for_unknown_type(self):
        registry = _make_same_type_registry()
        registry._agents.clear()
        with pytest.raises(KeyError):
            registry.get(ConceptType.DECISION)

    @pytest.mark.asyncio
    async def test_resolve_all_returns_empty_when_no_nodes(self):
        registry = _make_same_type_registry()
        rels, failed = await registry.resolve_all(source_nodes=[], per_source_targets={})
        assert rels == []
        assert failed == []

    @pytest.mark.asyncio
    async def test_resolve_all_partitions_by_type_and_collects_results(self):
        from seshat.agents.resolution.base import ResolvedRelationship
        from seshat.models.enums import RelationshipType

        registry = _make_same_type_registry()

        rel = MagicMock(spec=ResolvedRelationship)
        rel.rel_type = RelationshipType.SUPERSEDES

        for agent in registry._agents.values():
            agent.resolve = AsyncMock(return_value=([rel], []))

        decision_src = make_node("src1", title="Use PostgreSQL")
        decision_tgt = make_node("tgt1", title="Use MySQL")

        rels, failed = await registry.resolve_all(
            source_nodes=[decision_src],
            per_source_targets={decision_src.id: [decision_tgt]},
        )

        assert len(rels) == 1
        assert rels[0].rel_type == RelationshipType.SUPERSEDES
        assert failed == []

    @pytest.mark.asyncio
    async def test_resolve_all_returns_empty_when_no_same_type_targets(self):
        registry = _make_same_type_registry()

        for agent in registry._agents.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_node = make_node("n1")
        risk_node = make_node("n2")
        risk_node = risk_node.model_copy(update={"type": ConceptType.RISK})

        rels, failed = await registry.resolve_all(
            source_nodes=[decision_node],
            per_source_targets={decision_node.id: [risk_node]},
        )

        assert rels == []
        assert failed == []


class TestCrossTypeResolutionRegistry:
    def test_get_returns_agent_for_known_pair(self):
        registry = _make_cross_type_registry()
        agent = registry.get(ConceptType.DECISION, ConceptType.RISK)
        assert agent is not None

    def test_get_raises_for_unknown_pair(self):
        registry = _make_cross_type_registry()
        with pytest.raises(KeyError):
            registry.get(ConceptType.DECISION, ConceptType.DECISION)

    @pytest.mark.asyncio
    async def test_resolve_all_returns_empty_when_no_nodes(self):
        registry = _make_cross_type_registry()
        rels, failed = await registry.resolve_all(source_nodes=[], per_source_targets={})
        assert rels == []
        assert failed == []

    @pytest.mark.asyncio
    async def test_resolve_all_dispatches_to_matching_pair_agent(self):
        from seshat.agents.resolution.base import ResolvedRelationship
        from seshat.models.enums import RelationshipType

        registry = _make_cross_type_registry()

        rel = MagicMock(spec=ResolvedRelationship)
        rel.rel_type = RelationshipType.MITIGATES

        for agent in registry._agents_mapping.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_to_risk_agent = registry._agents_mapping[(ConceptType.DECISION, ConceptType.RISK)]
        decision_to_risk_agent.resolve = AsyncMock(return_value=([rel], []))

        decision_node = make_node("src1")
        risk_node = make_node("tgt1")
        risk_node = risk_node.model_copy(update={"type": ConceptType.RISK})

        rels, failed = await registry.resolve_all(
            source_nodes=[decision_node],
            per_source_targets={decision_node.id: [risk_node]},
        )

        assert len(rels) == 1
        assert rels[0].rel_type == RelationshipType.MITIGATES
        assert failed == []

    @pytest.mark.asyncio
    async def test_resolve_all_skips_pairs_with_no_matching_nodes(self):
        registry = _make_cross_type_registry()

        for agent in registry._agents_mapping.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_node = make_node("n1")

        rels, failed = await registry.resolve_all(
            source_nodes=[decision_node],
            per_source_targets={decision_node.id: [decision_node]},
        )

        assert rels == []
        assert failed == []
        for agent in registry._agents_mapping.values():
            agent.resolve.assert_not_called()


class TestGlobalSemaphoreForwarding:
    @pytest.mark.asyncio
    async def test_global_sem_forwarded_to_same_type_agents(self):
        registry = _make_same_type_registry()
        sem = asyncio.Semaphore(1)

        for agent in registry._agents.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_src = make_node("src1")
        decision_tgt = make_node("tgt1")

        await registry.resolve_all(
            source_nodes=[decision_src],
            per_source_targets={decision_src.id: [decision_tgt]},
            global_sem=sem,
        )

        decision_agent = registry._agents[ConceptType.DECISION]
        call_kwargs = decision_agent.resolve.call_args
        assert call_kwargs.args[2] is sem or call_kwargs.kwargs.get("global_sem") is sem

    @pytest.mark.asyncio
    async def test_global_sem_forwarded_to_cross_type_agents(self):
        registry = _make_cross_type_registry()
        sem = asyncio.Semaphore(1)

        for agent in registry._agents_mapping.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_node = make_node("src1")
        risk_node = make_node("tgt1")
        risk_node = risk_node.model_copy(update={"type": ConceptType.RISK})

        await registry.resolve_all(
            source_nodes=[decision_node],
            per_source_targets={decision_node.id: [risk_node]},
            global_sem=sem,
        )

        decision_to_risk = registry._agents_mapping[(ConceptType.DECISION, ConceptType.RISK)]
        call_kwargs = decision_to_risk.resolve.call_args
        assert call_kwargs.args[2] is sem or call_kwargs.kwargs.get("global_sem") is sem
