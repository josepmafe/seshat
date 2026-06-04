from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.common import log_breakdown_artifact
from seshat.eval.gate import upsert_gate
from seshat.eval.resolution.corpus_loader import build_kb_nodes, load_corpus
from seshat.eval.resolution.scorers import scorer
from seshat.models.enums import ConceptType
from seshat.models.nodes import ResolutionResult

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.config.settings import EvalConfig
    from seshat.eval.models import GateResult, ResolutionCorpusExample
    from seshat.models.nodes import KBNode
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator


class ResolutionEvalRunner:
    def __init__(
        self,
        orchestrator: ExtractionOrchestrator,
        config: EvalConfig,
        model_id: str | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._config = config
        self._model_id = model_id
        self._kb_nodes: dict[str, dict[str, KBNode]] = {}
        self._slug_maps: dict[str, dict[str, UUID]] = {}

    async def run(self, tag_filter: dict[str, str | list[str]] | None = None) -> GateResult:
        mlflow.autolog(disable=True)
        examples = load_corpus(self._config.resolution_corpus_dir, tag_filter=tag_filter)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="resolution-no-corpus")

        for ex in examples:
            kb_nodes, slug_map = build_kb_nodes(ex)
            self._kb_nodes[ex.corpus_id] = kb_nodes
            self._slug_maps[ex.corpus_id] = slug_map

        result_cache, touched = await self._run_all_predictions(examples)

        def _predict(corpus_id: str) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not found in result cache — mlflow unpacking mismatch")
            return {"relationships": [r.model_dump(mode="json") for r in result_cache[corpus_id].relationships]}

        df = _build_dataframe(examples, self._slug_maps)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=self._model_id)

        run_id = eval_result.run_id
        resolution_metrics = _aggregate_metrics(eval_result)
        self._log_breakdown(eval_result, examples, result_cache, run_id)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            resolution_metrics=resolution_metrics,
        )
        mlflow.log_metrics({**resolution_metrics, "gate.passed": float(gate.passed)}, run_id=run_id)
        if tag_filter:
            mlflow.log_params({f"tag_filter.{k}": str(v) for k, v in tag_filter.items()}, run_id=run_id)

        sweep_stale_entries(
            self._config.resolution_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
        )
        return gate

    async def _run_all_predictions(
        self, examples: list[ResolutionCorpusExample]
    ) -> tuple[dict[str, ResolutionResult], set[Path]]:
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)

        async def _run_one(ex: ResolutionCorpusExample) -> tuple[str, ResolutionResult, Path]:
            kb_nodes = self._kb_nodes[ex.corpus_id]
            source_nodes = [kb_nodes[n.id] for n in ex.source_nodes]
            kb_target_nodes = [kb_nodes[n.id] for n in ex.kb_nodes]
            per_source_targets: dict[UUID, list[KBNode]] = {src.id: kb_target_nodes for src in source_nodes}

            agent_hash = self._orchestrator._resolution_registry.fingerprint_for_types(
                source_types={n.type for n in source_nodes}, target_types={n.type for n in kb_target_nodes}
            )
            cache_fp = build_cache_fp(self._config.resolution_cache_dir, ex, agent_hash=agent_hash)

            async with sem:
                result, used = await read_or_run(
                    cache_fp,
                    ResolutionResult,
                    self._orchestrator._run_resolution(source_nodes, per_source_targets, job_id=ex.corpus_id),
                )
            return ex.corpus_id, result, used

        triples = await asyncio.gather(*(_run_one(ex) for ex in examples))
        results = {corpus_id: result for corpus_id, result, _ in triples}
        touched = {used for _, _, used in triples}
        return results, touched

    def _log_breakdown(
        self,
        eval_result: EvaluationResult,
        examples: list[ResolutionCorpusExample],
        result_cache: dict[str, ResolutionResult],
        run_id: str,
    ) -> None:
        log_breakdown_artifact(_build_breakdown(eval_result, examples, result_cache, self._slug_maps), run_id)


def _build_dataframe(examples: list[ResolutionCorpusExample], slug_maps: dict[str, dict[str, UUID]]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        uuid_str_map = {k: str(v) for k, v in slug_maps[ex.corpus_id].items()}
        slug_to_type = {n.id: n.type.value for n in ex.source_nodes + ex.kb_nodes}
        rows.append(
            {
                "inputs": {"corpus_id": ex.corpus_id},
                "expectations": {
                    "expected_relations": [
                        {"source": r.source, "target": r.target, "rel_type": r.rel_type.value}
                        for r in ex.expected_relations
                    ],
                    "slug_to_uuid": uuid_str_map,
                    "slug_to_type": slug_to_type,
                },
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    """Flatten per-type precision/recall into dotted keys: '{ctype}.precision', '{ctype}.recall'."""
    result: dict[str, float] = {}
    for ctype in ConceptType:
        p = eval_result.metrics.get(f"{ctype}.precision/mean")
        r = eval_result.metrics.get(f"{ctype}.recall/mean")
        if p is not None:
            result[f"{ctype}.precision"] = float(p)
        if r is not None:
            result[f"{ctype}.recall"] = float(r)
    return result


def _build_breakdown(
    eval_result: EvaluationResult,
    examples: list[ResolutionCorpusExample],
    result_cache: dict[str, ResolutionResult],
    slug_maps: dict[str, dict[str, UUID]],
) -> dict:
    assert eval_result.result_df is not None
    breakdown: dict[str, dict] = {}
    for ex, (_, row) in zip(examples, eval_result.result_df.iterrows(), strict=True):
        scores: dict[ConceptType, dict[str, float | None]] = {}
        for ctype in ConceptType:
            p = row.get(f"{ctype}.precision/value")
            r = row.get(f"{ctype}.recall/value")
            if pd.isna(p) and pd.isna(r):
                continue
            scores[ctype] = {
                "precision": float(p) if not pd.isna(p) else None,
                "recall": float(r) if not pd.isna(r) else None,
            }

        uuid_to_slug = {v: k for k, v in slug_maps[ex.corpus_id].items()}
        breakdown[ex.corpus_id] = {
            "tags": ex.tags,
            "group": "same_type" if _is_same_type(ex) else "cross_type",
            "scores": scores,
            "expected": [
                {"source": r.source, "target": r.target, "rel_type": r.rel_type.value} for r in ex.expected_relations
            ],
            "predicted": [
                {
                    "source": uuid_to_slug.get(r.source_id, str(r.source_id)),
                    "target": uuid_to_slug.get(r.target_id, str(r.target_id)),
                    "rel_type": r.rel_type,
                }
                for r in result_cache[ex.corpus_id].relationships
            ],
        }
    return breakdown


def _is_same_type(example: ResolutionCorpusExample) -> bool:
    source_types = {n.type for n in example.source_nodes}
    kb_types = {n.type for n in example.kb_nodes}
    return bool(source_types & kb_types)
