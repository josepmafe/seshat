from __future__ import annotations

import json
import tempfile
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.eval.gate import upsert_gate
from seshat.eval.resolution_corpus_loader import build_kb_nodes_with_slug_map, load_resolution_corpus
from seshat.eval.scorers import resolution_scorer

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.config.settings import EvalConfig
    from seshat.eval.models import GateResult, ResolutionCorpusExample
    from seshat.models.nodes import KBNode, ResolutionResult
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator


class ResolutionEvalRunner:
    def __init__(
        self,
        orchestrator: ExtractionOrchestrator,
        config: EvalConfig,
    ) -> None:
        self._orchestrator = orchestrator
        self._config = config

    async def run(self) -> GateResult:
        mlflow.set_tracking_uri(self._config.observability.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.observability.mlflow_experiment_name)

        examples = load_resolution_corpus(self._config.resolution_corpus_dir)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="resolution-no-corpus")

        result_cache = await self._run_all_predictions(examples)

        rows = []
        for ex in examples:
            _, slug_to_uuid = build_kb_nodes_with_slug_map(ex)
            uuid_str_map = {k: str(v) for k, v in slug_to_uuid.items()}
            rows.append(
                {
                    "inputs": {"corpus_id": ex.corpus_id},
                    "expectations": {
                        "expected_relations": [
                            {"source": r.source, "target": r.target, "rel_type": r.rel_type.value}
                            for r in ex.expected_relations
                        ],
                        "slug_to_uuid": uuid_str_map,
                    },
                }
            )

        def _predict(corpus_id: str) -> dict:
            result = result_cache[corpus_id]
            return {"relationships": [r.model_dump(mode="json") for r in result.relationships]}

        df = pd.DataFrame(rows)
        eval_result = mlflow.genai.evaluate(
            data=df,
            predict_fn=_predict,
            scorers=[resolution_scorer],
        )

        run_id = eval_result.run_id
        resolution_metrics = _aggregate_metrics(eval_result)
        self._log_breakdown(examples, result_cache, run_id)

        return upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            resolution_metrics=resolution_metrics,
        )

    async def _run_all_predictions(self, examples: list[ResolutionCorpusExample]) -> dict[str, ResolutionResult]:
        cache: dict[str, ResolutionResult] = {}
        for ex in examples:
            kb_nodes, _ = build_kb_nodes_with_slug_map(ex)
            source_nodes = [kb_nodes[n.id] for n in ex.source_nodes]
            kb_target_nodes = [kb_nodes[n.id] for n in ex.kb_nodes]
            per_source_targets: dict[UUID, list[KBNode]] = {src.id: kb_target_nodes for src in source_nodes}
            cache[ex.corpus_id] = await self._orchestrator._run_resolution(
                source_nodes, per_source_targets, job_id="eval"
            )
        return cache

    def _log_breakdown(
        self,
        examples: list[ResolutionCorpusExample],
        result_cache: dict[str, ResolutionResult],
        run_id: str,
    ) -> None:
        breakdown = {
            ex.corpus_id: {
                "predicted": [r.model_dump(mode="json") for r in result_cache[ex.corpus_id].relationships],
                "expected": [
                    {"source": r.source, "target": r.target, "rel_type": r.rel_type.value}
                    for r in ex.expected_relations
                ],
            }
            for ex in examples
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(breakdown, f, indent=2)
            breakdown_path = f.name

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(breakdown_path, artifact_path="eval")


def _aggregate_metrics(eval_result) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("precision", "recall", "f1"):
        v = eval_result.metrics.get(f"{metric}/mean")
        if v is not None:
            result[metric] = float(v)
    return result
