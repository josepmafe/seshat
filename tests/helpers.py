from datetime import date
from uuid import NAMESPACE_DNS, uuid5

from seshat.models.enums import ConceptType, IngestionSource, NodeStatus
from seshat.models.nodes import KBNode, NodeMetadata
from seshat.models.quote_anchor import QuoteAnchor


def make_node(
    node_id: str = "n1",
    title: str = "Use PostgreSQL",
    confidence: float = 0.9,
    team: str | None = None,
    type: ConceptType = ConceptType.DECISION,
    description: str = "Team decided to use PostgreSQL.",
    status: NodeStatus = NodeStatus.APPROVED,
) -> KBNode:
    return KBNode(
        id=uuid5(NAMESPACE_DNS, node_id),
        type=type,
        title=title,
        description=description,
        confidence=confidence,
        quote_anchors=[QuoteAnchor(transcript_file="test.txt", char_start=0, char_end=22)],
        status=status,
        metadata=NodeMetadata(
            job_id="job-1",
            meeting_date=date(2026, 4, 21),
            ingestion_source=IngestionSource.JOB,
            team=team,
        ),
    )
