from seshat.config.settings import ConfidenceWeights
from seshat.models.nodes import ConfidenceBreakdown


def compute_confidence(
    *,
    verification: float | None,
    heuristics: float,
    weights: ConfidenceWeights,
    verification_enabled: bool = False,
) -> ConfidenceBreakdown:
    signals: list[tuple[float, float]] = [
        (weights.heuristics, heuristics),
    ]

    if verification is not None:
        signals.append((weights.verification, verification))

    total_weight = sum(w for w, _ in signals)
    final = sum(w * s for w, s in signals) / total_weight if total_weight > 0.0 else 0.0

    return ConfidenceBreakdown(
        verification_enabled=verification_enabled,
        verification=verification,
        heuristics=heuristics,
        final=final,
    )
