from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field

from seshat.models.base import SeshatModel
from seshat.models.enums import (
    ApprovalMethod,
    ConceptType,
    IngestionSource,
    NodeState,
    NodeStatus,
    RelationshipType,
)
from seshat.models.quote_anchor import QuoteAnchor


class KBRelationship(SeshatModel):
    source_id: UUID
    target_id: UUID
    rel_type: RelationshipType
    job_id: str
    created_at: datetime = Field(description="UTC timestamp when this relationship was written.")


class ConfidenceBreakdown(SeshatModel):
    verification: float | None = Field(
        default=None, ge=0, le=1, description="Score from the verification LLM call; None means disabled verification."
    )
    # heuristics is always active — never None; guards against divide-by-zero in the scorer.
    heuristics: float = Field(
        ge=0, le=1, description="Heuristic signal (always present); used as fallback when other signals are absent."
    )
    final: float = Field(ge=0, le=1, description="Weighted composite score in [0, 1].")


class NodeMetadata(SeshatModel):
    job_id: str
    meeting_date: date | None = None
    participants: list[str] | None = None
    ingestion_source: IngestionSource = Field(
        default=IngestionSource.JOB, description="Whether this node came from a job or an init run."
    )
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = Field(default=None, description="UTC timestamp of approval.")
    approval_method: ApprovalMethod | None = None
    corrected_by: str | None = None
    corrected_at: datetime | None = Field(default=None, description="UTC timestamp of the last correction.")
    confidence_breakdown: ConfidenceBreakdown | None = Field(
        default=None, description="Per-signal confidence breakdown; persisted for UI display."
    )
    concept_fields: dict[str, Any] | None = Field(
        default=None, description="Type-specific fields from extraction (e.g. assignee, due, rationale)."
    )


class KBNode(SeshatModel):
    id: UUID = Field(default_factory=uuid4)
    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    type: ConceptType
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    quote_anchors: list[QuoteAnchor] = Field(
        default_factory=list,
        description="Anchored positions of source quotes within the transcript blob.",
    )
    status: NodeStatus
    state: NodeState = NodeState.CURRENT
    metadata: NodeMetadata


class ExtractionResult(SeshatModel):
    job_id: str
    nodes: list[KBNode]
    confidence_breakdowns: dict[UUID, ConfidenceBreakdown]
    failed_concept_types: list[ConceptType] = Field(
        default_factory=list,
        description="Concept types whose extraction task failed entirely; empty means all types completed.",
    )


class ResolutionResult(SeshatModel):
    job_id: str
    relationships: list[KBRelationship]
