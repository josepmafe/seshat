from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

from seshat.models.enums import ConceptType, RelationshipType

# ── Identification corpus ────────────────────────────────────────────────────


class IdentificationCorpusNode(BaseModel):
    quote: str  # ground-truth quote used by span-overlap matcher
    type: ConceptType
    title: str
    description: str
    extra_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific expected field values (assignee, due, rationale, etc.).",
    )


class IdentificationCorpusExample(BaseModel):
    corpus_id: str
    transcript: str
    expected_nodes: list[IdentificationCorpusNode]
    tags: dict[str, Any] = Field(default_factory=dict)


# ── Resolution corpus ────────────────────────────────────────────────────────


class ResolutionCorpusNode(BaseModel):
    id: str  # human-readable slug — local cross-reference key only
    type: ConceptType
    title: str
    description: str
    quote: str


class ResolutionCorpusRelation(BaseModel):
    source: str  # slug
    target: str  # slug
    rel_type: RelationshipType


class ResolutionCorpusExample(BaseModel):
    corpus_id: str
    description: str
    source_nodes: list[ResolutionCorpusNode]
    kb_nodes: list[ResolutionCorpusNode]
    expected_relations: list[ResolutionCorpusRelation]
    tags: dict[str, Any] = Field(default_factory=dict)


# ── Retrieval corpus ─────────────────────────────────────────────────────────


class RetrievalCorpusNode(BaseModel):
    id: str  # slug
    type: ConceptType
    title: str
    description: str
    quote: str


class RetrievalCorpusExample(BaseModel):
    corpus_id: str
    description: str
    query_node: RetrievalCorpusNode
    candidate_nodes: list[RetrievalCorpusNode]
    expected_relevant_ids: list[str]  # slugs from candidate_nodes


# ── Retrieval result ─────────────────────────────────────────────────────────


class RetrievalResult(BaseModel):
    retrieved_ids: list[str]


class RetrievalScoredResult(BaseModel):
    """Slug-keyed search results with scores — used by RetrievalMetaScorer's file cache."""

    results: list[tuple[str, float]]  # (slug, score) pairs, sorted desc by score


# ── Gate result ──────────────────────────────────────────────────────────────


class MetricEntry(BaseModel):
    value: float
    passed: bool


class GateResult(BaseModel):
    run_id: str
    timestamp: str = ""
    # dotted keys: "{ctype}.precision", "{ctype}.recall", "{ctype}.spurious_rate"
    identification_metrics: dict[str, MetricEntry] | None = None
    # dotted keys: "{ctype}.precision", "{ctype}.recall"
    resolution_metrics: dict[str, MetricEntry] | None = None
    # keys: "recall_at_5", "precision_at_5"
    retrieval_metrics: dict[str, MetricEntry] | None = None
    # keys: "precision", "recall"
    verification_metrics: dict[str, MetricEntry] | None = None
    # keys: "group_hit_rate" (gated), "exact_match" (logged, not gated)
    grouping_metrics: dict[str, MetricEntry] | None = None
    validation_hash: str = ""

    @computed_field
    @property
    def passed(self) -> bool:
        if self._all_metrics_are_none():
            return False
        return (
            self._identification_passes()
            and self._resolution_passes()
            and self._retrieval_passes()
            and self._grouping_passes()
            and self._verification_passes()
        )

    def model_post_init(self, __context: object) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
        self.validation_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = self.model_dump(exclude={"passed", "validation_hash"})
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def _all_metrics_are_none(self) -> bool:
        return (
            self.identification_metrics is None
            and self.resolution_metrics is None
            and self.retrieval_metrics is None
            and self.verification_metrics is None
            and self.grouping_metrics is None
        )

    def _identification_passes(self) -> bool:
        if self.identification_metrics is None:
            return True
        return all(e.passed for e in self.identification_metrics.values())

    def _resolution_passes(self) -> bool:
        if self.resolution_metrics is None:
            return True
        return all(e.passed for e in self.resolution_metrics.values())

    def _retrieval_passes(self) -> bool:
        if self.retrieval_metrics is None:
            return True
        return all(e.passed for e in self.retrieval_metrics.values())

    def _grouping_passes(self) -> bool:
        if self.grouping_metrics is None:
            return True
        return all(e.passed for e in self.grouping_metrics.values())

    def _verification_passes(self) -> bool:
        if self.verification_metrics is None:
            return True
        return all(e.passed for e in self.verification_metrics.values())
