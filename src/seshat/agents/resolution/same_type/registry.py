from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from seshat.agents.resolution.same_type.action_item import ActionItemResolutionAgent
from seshat.agents.resolution.same_type.decision import DecisionResolutionAgent
from seshat.agents.resolution.same_type.open_question import OpenQuestionResolutionAgent
from seshat.agents.resolution.same_type.risk import RiskResolutionAgent
from seshat.models.enums import ConceptType
from seshat.utils.log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel

    from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, ResolvedRelationship
    from seshat.config.settings import ResolutionLLMConfig
    from seshat.models.nodes import FailedResolutionSource, KBNode


class SameTypeResolutionRegistry:
    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig) -> None:
        self._agents: dict[ConceptType, BaseSameTypeResolutionAgent] = {
            ConceptType.DECISION: DecisionResolutionAgent(llm, config),
            ConceptType.RISK: RiskResolutionAgent(llm, config),
            ConceptType.ACTION_ITEM: ActionItemResolutionAgent(llm, config),
            ConceptType.OPEN_QUESTION: OpenQuestionResolutionAgent(llm, config),
        }

    def get(self, concept_type: ConceptType) -> BaseSameTypeResolutionAgent:
        agent = self._agents.get(concept_type)
        if agent is None:
            raise KeyError(f"No resolution agent registered for {concept_type}")
        return agent

    async def resolve_all(
        self,
        source_nodes: list[KBNode],
        per_source_targets: dict[UUID, list[KBNode]],
        global_sem: asyncio.Semaphore | None = None,
    ) -> tuple[list[ResolvedRelationship], list[FailedResolutionSource]]:
        sources_by_type: dict[ConceptType, list[KBNode]] = {}
        for node in source_nodes:
            sources_by_type.setdefault(node.type, []).append(node)

        concept_types, tasks = [], []
        for ct, sources, scoped in self._iter_active_types(sources_by_type, per_source_targets):
            concept_types.append(ct)
            tasks.append(self._agents[ct].resolve(sources, scoped, global_sem))

        if not tasks:
            return [], []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        resolved: list[ResolvedRelationship] = []
        failed: list[FailedResolutionSource] = []
        for ct, result in zip(concept_types, results, strict=True):
            if isinstance(result, Exception):
                logger.error("Same-type resolution failed for %s: %s", ct, result)
                continue

            assert isinstance(result, tuple)
            rels, fails = result
            resolved.extend(rels)
            failed.extend(fails)
        return resolved, failed

    def _iter_active_types(
        self,
        sources_by_type: dict[ConceptType, list[KBNode]],
        per_source_targets: dict[UUID, list[KBNode]],
    ) -> Iterator[tuple[ConceptType, list[KBNode], dict[UUID, list[KBNode]]]]:
        for ct, sources in sources_by_type.items():
            if ct not in self._agents:
                continue
            yield ct, sources, _scope_targets(sources, per_source_targets, ct)


def _scope_targets(
    sources: list[KBNode],
    per_source_targets: dict[UUID, list[KBNode]],
    target_type: ConceptType,
) -> dict[UUID, list[KBNode]]:
    targets = {}
    for src in sources:
        scoped = [t for t in per_source_targets.get(src.id, []) if t.type == target_type]
        targets[src.id] = scoped
    return targets
