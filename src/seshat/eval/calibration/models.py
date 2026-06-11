from __future__ import annotations

from pydantic import BaseModel

from seshat.models.enums import ConceptType


class TypeMetrics(BaseModel):
    precision: float
    recall: float
    f1: float


class TypePC(BaseModel):
    precision_approved: float  # TP_auto / (TP_auto + FP_auto); NaN-safe: 1.0 when no nodes approved
    coverage: float  # (TP_auto + FP_auto) / total_extracted; 0.0 when nothing extracted


class IdentificationSweepPoint(BaseModel):
    threshold: float
    precision_approved: float  # aggregate across all types
    coverage: float  # aggregate across all types
    per_type: dict[ConceptType, TypePC]


class IdentificationSweepResult(BaseModel):
    points: list[IdentificationSweepPoint]
    suggested_threshold: float  # argmax coverage s.t. precision_approved >= p_target; ties → lower threshold


class RetrievalSweepPoint(BaseModel):
    threshold: float
    recall_at_5: float
    precision_at_5: float
    macro_f2: float


class RetrievalSweepResult(BaseModel):
    points: list[RetrievalSweepPoint]  # sorted ascending by threshold
    suggested_threshold: float  # argmax(macro_f2); ties → lower threshold
