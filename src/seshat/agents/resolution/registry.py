from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from seshat.agents.resolution.cross_type.registry import CrossTypeResolutionRegistry
from seshat.agents.resolution.same_type.registry import SameTypeResolutionRegistry

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel

    from seshat.agents.resolution.base import ResolvedRelationship
    from seshat.config.settings import ResolutionLLMConfig
    from seshat.models.nodes import FailedResolutionSource, KBNode


class ResolutionRegistry:
    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig) -> None:
        self._same_type = SameTypeResolutionRegistry(llm, config)
        self._cross_type = CrossTypeResolutionRegistry(llm, config)

    async def resolve_all(
        self,
        source_nodes: list[KBNode],
        per_source_targets: dict[UUID, list[KBNode]],
        semaphore: asyncio.Semaphore | None = None,
    ) -> tuple[list[ResolvedRelationship], list[FailedResolutionSource]]:
        (same_rels, same_failed), (cross_rels, cross_failed) = await asyncio.gather(
            self._same_type.resolve_all(source_nodes, per_source_targets, semaphore),
            self._cross_type.resolve_all(source_nodes, per_source_targets, semaphore),
        )
        return same_rels + cross_rels, same_failed + cross_failed
