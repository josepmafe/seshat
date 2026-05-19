from uuid import uuid4

import pytest

from seshat.agents.identification.registry import IdentificationAgentRegistry
from seshat.agents.resolution.registry import ResolutionRegistry
from seshat.blob_store.s3_store import S3BlobStore
from seshat.config.settings import BlobStoreConfig, ExtractionConfig, RAGConfig
from seshat.models.enums import ConceptType, NodeStatus
from seshat.pipeline.extraction.node_retriever import NodeRetriever
from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
from tests.helpers import make_doc, make_node
from tests.integration.agents.conftest import _cheap_llm_config, _make_cheap_llm
from tests.integration.conftest import (
    LOCALSTACK_REGION,
    LOCALSTACK_TEST_BUCKET,
    SKIP_IF_NO_EMBEDDINGS_API,
    SKIP_IF_NO_LOCALSTACK,
    SKIP_IF_NO_POSTGRES,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.agents,
    pytest.mark.embedding,
    SKIP_IF_NO_POSTGRES,
    SKIP_IF_NO_EMBEDDINGS_API,
    SKIP_IF_NO_LOCALSTACK,
]

_TRANSCRIPT = """
We've reviewed the options. We're going to use PostgreSQL for the user database — it has the best JSON support and the
team is familiar with it.

Agreed. One risk though: we haven't done a migration rehearsal yet, so there's a chance of data corruption if the
script fails.

Fair point. Let's make sure Sergio writes the migration script by Friday.

What cloud provider are we actually deploying to? That's still not decided.
"""


@pytest.fixture
async def blob_store(localstack_s3_url):
    config = BlobStoreConfig(
        bucket=LOCALSTACK_TEST_BUCKET,
        region=LOCALSTACK_REGION,
        endpoint_url=localstack_s3_url,
    )
    store = S3BlobStore(config)
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def extraction_config():
    llm_cfg = _cheap_llm_config()
    return ExtractionConfig(identification=llm_cfg, result_cache_enabled=False)


@pytest.fixture
def orchestrator(kb_store, vector_store, blob_store, extraction_config):
    llm = _make_cheap_llm()
    rag = NodeRetriever(RAGConfig(), kb_store, vector_store)
    return ExtractionOrchestrator(
        config=extraction_config,
        identification_registry=IdentificationAgentRegistry(llm, extraction_config),
        resolution_registry=ResolutionRegistry(llm, extraction_config.resolution),
        node_retriever=rag,
        kb_store=kb_store,
        blob_store=blob_store,
    )


class TestExtractionOrchestrator:
    async def test_run_identification_returns_nodes_with_confidence(self, orchestrator, blob_store):
        blob_key = f"transcripts/{uuid4()}.txt"
        await blob_store.put(blob_key, _TRANSCRIPT.encode())
        job_id = str(uuid4())

        result = await orchestrator.run_identification(make_doc(blob_key), job_id)

        assert result.job_id == job_id
        assert len(result.nodes) >= 1
        for node in result.nodes:
            assert node.metadata.job_id == job_id
            assert node.metadata.confidence_breakdown is not None
            assert 0.0 <= node.confidence <= 1.0

    async def test_run_resolution_returns_relationships(self, orchestrator, kb_store, vector_store, blob_store):
        blob_key = f"transcripts/{uuid4()}.txt"
        await blob_store.put(blob_key, _TRANSCRIPT.encode())
        job_id = str(uuid4())
        seed_job_id = str(uuid4())

        old_decision = make_node(
            node_id="orch-old-decision",
            title="Use MySQL for the user database",
            description="The team previously decided to use MySQL for the user database.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )
        new_decision = make_node(
            node_id="orch-new-decision",
            title="Use PostgreSQL for the user database",
            description="The team decided to switch to PostgreSQL for better JSON support.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )

        # old_decision is from a prior job — a candidate target, not a source for this run
        await kb_store.write_node(
            old_decision.model_copy(
                update={"metadata": old_decision.metadata.model_copy(update={"job_id": seed_job_id})}
            )
        )
        await vector_store.upsert(
            str(old_decision.id),
            f"{old_decision.title} {old_decision.description}",
            {"node_type": old_decision.type.value, "confidence": old_decision.confidence, "job_id": seed_job_id},
        )

        # new_decision is from the current job — the source node for resolution
        await kb_store.write_node(
            new_decision.model_copy(update={"metadata": new_decision.metadata.model_copy(update={"job_id": job_id})})
        )
        await vector_store.upsert(
            str(new_decision.id),
            f"{new_decision.title} {new_decision.description}",
            {"node_type": new_decision.type.value, "confidence": new_decision.confidence, "job_id": job_id},
        )

        result = await orchestrator.run_resolution(make_doc(blob_key), job_id)

        assert result.job_id == job_id
        assert len(result.relationships) >= 1

    async def test_identification_then_resolution_produces_consistent_job_id(
        self, orchestrator, kb_store, vector_store, blob_store
    ):
        """Identification nodes stored by identification are findable by run_resolution for the same job_id."""
        blob_key = f"transcripts/{uuid4()}.txt"
        await blob_store.put(blob_key, _TRANSCRIPT.encode())
        job_id = str(uuid4())

        identification_result = await orchestrator.run_identification(make_doc(blob_key), job_id)
        assert len(identification_result.nodes) >= 1

        for node in identification_result.nodes:
            await vector_store.upsert(
                str(node.id),
                f"{node.title} {node.description}",
                {"node_type": node.type.value, "confidence": node.confidence, "job_id": job_id},
            )
            await kb_store.write_node(node)

        resolution_result = await orchestrator.run_resolution(make_doc(blob_key), job_id)

        assert resolution_result.job_id == job_id
