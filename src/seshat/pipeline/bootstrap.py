from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.agents.identification.registry import IdentificationAgentRegistry
from seshat.agents.resolution.registry import ResolutionRegistry
from seshat.pipeline.extraction.node_retriever import NodeRetriever
from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
from seshat.pipeline.llm_factory import get_identification_llm, get_resolution_llm

if TYPE_CHECKING:
    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.vector_store.base_store import AbstractVectorStore


def build_orchestrator(
    config: SeshatConfig,
    kb_store: PostgresKBStore,
    vector_store: AbstractVectorStore,
    blob_store: S3BlobStore,
) -> ExtractionOrchestrator:
    identification_llm = get_identification_llm(config)
    resolution_llm = get_resolution_llm(config)
    identification_registry = IdentificationAgentRegistry(llm=identification_llm, config=config.extraction)
    resolution_registry = ResolutionRegistry(llm=resolution_llm, config=config.extraction.resolution)
    node_retriever = NodeRetriever(rag_config=config.rag, kb_store=kb_store, vector_store=vector_store)
    return ExtractionOrchestrator(
        config=config.extraction,
        identification_registry=identification_registry,
        resolution_registry=resolution_registry,
        node_retriever=node_retriever,
        kb_store=kb_store,
        blob_store=blob_store,
    )
