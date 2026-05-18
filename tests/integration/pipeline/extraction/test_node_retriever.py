import pytest

from seshat.config.settings import RAGConfig
from seshat.models.enums import ConceptType, NodeStatus
from seshat.pipeline.extraction.node_retriever import NodeRetriever
from tests.helpers import make_node
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API, SKIP_IF_NO_POSTGRES

pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.embedding,
    SKIP_IF_NO_POSTGRES,
    SKIP_IF_NO_EMBEDDINGS_API,
]


@pytest.fixture
def node_retriever(kb_store, vector_store) -> NodeRetriever:
    return NodeRetriever(RAGConfig(), kb_store, vector_store)


async def _seed_node(node, kb_store, vector_store, *, job_id: str | None = None) -> None:
    stored = (
        node
        if job_id is None
        else node.model_copy(update={"metadata": node.metadata.model_copy(update={"job_id": job_id})})
    )
    await kb_store.write_node(stored)
    metadata = {"node_type": node.type.value, "confidence": node.confidence}
    if job_id is not None:
        metadata["job_id"] = job_id
    await vector_store.upsert(str(node.id), f"{node.title} {node.description}", metadata)


class TestNodeRetrieverRetrieveCandidates:
    async def test_returns_seeded_node_for_similar_query(self, node_retriever, kb_store, vector_store):
        seeded = make_node(
            node_id="rag-seed",
            title="Use PostgreSQL for the user database",
            description="The team agreed to use PostgreSQL v15 due to its JSON support and performance.",
            status=NodeStatus.APPROVED,
        )
        await _seed_node(seeded, kb_store, vector_store)

        query_node = make_node(
            node_id="rag-query",
            title="Switch to PostgreSQL",
            description="We should adopt PostgreSQL as our primary database.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )

        results = await node_retriever.retrieve(query_node, "")

        assert any(r.id == seeded.id for r in results)

    async def test_orphan_vector_hit_is_silently_skipped(self, node_retriever, vector_store):
        orphan = make_node(
            node_id="rag-orphan",
            title="Use Kafka for the event bus",
            description="The team decided to use Kafka as the event streaming backbone.",
            status=NodeStatus.APPROVED,
        )
        # upsert into vector store only — no corresponding KB node written
        await vector_store.upsert(
            str(orphan.id),
            f"{orphan.title} {orphan.description}",
            {"node_type": orphan.type.value, "confidence": orphan.confidence},
        )

        query_node = make_node(
            node_id="rag-query-orphan",
            title="Event streaming with Kafka",
            description="We should adopt Kafka for event streaming.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )

        results = await node_retriever.retrieve(query_node, "")

        assert all(r.id != orphan.id for r in results)

    async def test_exclude_job_id_filters_nodes_from_same_job(self, node_retriever, kb_store, vector_store):
        current_job = "job-current"
        prior_job = "job-prior"

        current_node = make_node(
            node_id="rag-current",
            title="Use PostgreSQL for the user database",
            description="The team agreed to use PostgreSQL v15 due to its JSON support and performance.",
            status=NodeStatus.APPROVED,
        )
        prior_node = make_node(
            node_id="rag-prior",
            title="Use PostgreSQL as the primary database",
            description="Earlier decision to adopt PostgreSQL for the project.",
            status=NodeStatus.APPROVED,
        )
        await _seed_node(current_node, kb_store, vector_store, job_id=current_job)
        await _seed_node(prior_node, kb_store, vector_store, job_id=prior_job)

        query_node = make_node(
            node_id="rag-query-jobfilter",
            title="Switch to PostgreSQL",
            description="We should adopt PostgreSQL as our primary database.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )

        results = await node_retriever.retrieve(query_node, "", exclude_job_id=current_job)

        assert all(r.id != current_node.id for r in results)
