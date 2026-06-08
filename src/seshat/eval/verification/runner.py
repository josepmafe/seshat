from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.agents.verification import VerificationResult
from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.gate import upsert_gate
from seshat.eval.mlflow_logging import configure_trace_processors, log_eval_run_metadata, make_input_redactor
from seshat.eval.verification.corpus_loader import load_corpus
from seshat.eval.verification.scorers import scorer

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.agents.verification import VerificationAgent
    from seshat.config.eval_settings import EvalConfig
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.eval.models import GateResult
    from seshat.eval.verification.corpus_loader import VerificationCorpusExample


class VerificationEvalRunner:
    def __init__(self, agent: VerificationAgent, config: EvalConfig) -> None:
        self._agent = agent
        self._config = config

    async def run(self, tag_filter: CorpusTagFilter | None = None, model_id: str | None = None) -> GateResult:
        examples = load_corpus(self._config.verification_corpus_dir, tag_filter=tag_filter)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="verification-no-corpus")

        result_cache, touched = await self._run_all_predictions(examples)

        expected_by_key: dict[tuple[str, int], bool] = {
            (ex.corpus_id, i): node.expected_supported for ex in examples for i, node in enumerate(ex.nodes)
        }

        def _predict(corpus_id: str, node_index: int, _title: str, _description: str, _quote: str) -> dict:
            key = (corpus_id, node_index)
            if key not in result_cache:
                raise KeyError(f"key {key!r} not in result cache — mlflow unpacking mismatch")
            return {
                "supported": result_cache[key].supported,
                "expected_supported": expected_by_key[key],
            }

        configure_trace_processors(make_input_redactor(fields_to_exclude={"node_index"}))

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=model_id)

        run_id = eval_result.run_id
        verification_metrics = _aggregate_metrics(eval_result)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            verification_metrics=verification_metrics,
        )
        log_eval_run_metadata(
            run_id=run_id,
            harness="verification",
            gate_passed=gate.passed,
            corpus_dir=self._config.verification_corpus_dir,
            corpus_examples=examples,
            breakdown_artifact=_build_breakdown(examples, result_cache),
            tag_filter=tag_filter,
        )

        sweep_stale_entries(
            self._config.verification_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
        )
        return gate

    async def _run_all_predictions(
        self, examples: list[VerificationCorpusExample]
    ) -> tuple[dict[tuple[str, int], VerificationResult], set[Path]]:
        # Pre-populate before mlflow.genai.evaluate (sync) to avoid event-loop boundary issues.
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)
        agent_hash = self._agent.fingerprint()

        async def _run_one(ex: VerificationCorpusExample, i: int) -> tuple[tuple[str, int], VerificationResult, Path]:
            cache_fp = build_cache_fp(self._config.verification_cache_dir, ex, agent_hash=agent_hash, index=i)
            node = ex.nodes[i]

            async with sem:
                result, used = await read_or_run(
                    cache_fp,
                    VerificationResult,
                    self._agent.verify(
                        title=node.title,
                        description=node.description,
                        quote=node.quote,
                        transcript=ex.transcript,
                    ),
                )
            return (ex.corpus_id, i), result, used

        tasks = [_run_one(ex, i) for ex in examples for i in range(len(ex.nodes))]
        triples = await asyncio.gather(*tasks)
        results = {key: result for key, result, _ in triples}
        touched = {used for _, _, used in triples}
        return results, touched


def _build_dataframe(examples: list[VerificationCorpusExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        for i, node in enumerate(ex.nodes):
            rows.append(
                {
                    "inputs": {
                        "corpus_id": ex.corpus_id,
                        "node_index": i,
                        "_title": node.title,
                        "_description": node.description,
                        "_quote": node.quote,
                    },
                    "expectations": {"expected_supported": node.expected_supported},
                    "tags": {f"corpus.{k}": str(v) for k, v in ex.tags.items()},
                }
            )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    assert eval_result.result_df is not None
    for _, row in eval_result.result_df.iterrows():
        for k in counts:
            v = row.get(f"verification.{k}/value")
            if v is not None and not pd.isna(v):
                counts[k] += int(v)

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall}


def _build_breakdown(
    examples: list[VerificationCorpusExample],
    result_cache: dict[tuple[str, int], VerificationResult],
) -> dict:
    breakdown: dict = {}
    for ex in examples:
        nodes_out = []
        for i, node in enumerate(ex.nodes):
            result = result_cache.get((ex.corpus_id, i))
            nodes_out.append(
                {
                    "title": node.title,
                    "expected_supported": node.expected_supported,
                    "predicted_supported": result.supported if result else None,
                    "rationale": result.rationale if result else None,
                }
            )
        breakdown[ex.corpus_id] = {"tags": ex.tags, "nodes": nodes_out}
    return breakdown
