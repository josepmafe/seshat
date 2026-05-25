from __future__ import annotations

from pathlib import Path  # noqa: TC003

from seshat.eval.models import GateResult


def write_gate(result: GateResult, gate_path: Path) -> None:
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(result.model_dump_json(indent=2))


def read_gate(gate_path: Path) -> GateResult:
    return GateResult.model_validate_json(gate_path.read_text())


def upsert_gate(
    gate_path: Path,
    run_id: str,
    identification_metrics: dict[str, float] | None = None,
    resolution_metrics: dict[str, float] | None = None,
    retrieval_metrics: dict[str, float] | None = None,
) -> GateResult:
    """Update only the supplied metric blocks; carry over the rest from the existing file."""
    if gate_path.exists():
        existing = read_gate(gate_path)
        identification_metrics = identification_metrics or existing.identification_metrics
        resolution_metrics = resolution_metrics or existing.resolution_metrics
        retrieval_metrics = retrieval_metrics or existing.retrieval_metrics

    result = GateResult(
        run_id=run_id,
        identification_metrics=identification_metrics,
        resolution_metrics=resolution_metrics,
        retrieval_metrics=retrieval_metrics,
    )
    write_gate(result, gate_path)
    return result
