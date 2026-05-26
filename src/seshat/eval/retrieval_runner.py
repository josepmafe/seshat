from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.eval.gate import upsert_gate
from seshat.eval.retrieval_corpus_loader import build_retrieval_kb_nodes, load_retrieval_corpus
from seshat.eval.scorers import retrieval_scorer

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.config.settings import EvalConfig
    from seshat.eval.models import GateResult, RetrievalCorpusExample
    from seshat.models.nodes import KBNode
    from seshat.vector_store.base_store import AbstractVectorStore


class RetrievalEvalRunner:
    """Eval runner for the retrieval pass.

    The caller is responsible for passing a dedicated, empty vector store collection.
    Any pre-existing nodes in the collection will appear in search results and corrupt scores.
    """

    def __init__(
        self,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
    ) -> None:
        self._vs = vector_store
        self._config = config

    async def run(self) -> GateResult:
        mlflow.set_tracking_uri(self._config.observability.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.observability.mlflow_experiment_name)

        examples = load_retrieval_corpus(self._config.retrieval_corpus_dir)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="retrieval-no-corpus")

        # Build UUID maps once — build_retrieval_kb_nodes calls uuid4() so must not be called twice.
        example_nodes: dict[str, tuple[KBNode, list[KBNode], dict[str, UUID]]] = {
            ex.corpus_id: build_retrieval_kb_nodes(ex) for ex in examples
        }

        result_cache = await self._run_all_predictions(examples, example_nodes)

        rows = []
        for ex in examples:
            _, _, slug_map = example_nodes[ex.corpus_id]
            expected_uuids = [str(slug_map[s]) for s in ex.expected_relevant_ids if s in slug_map]
            rows.append(
                {
                    "inputs": {"corpus_id": ex.corpus_id},
                    "expectations": {"expected_relevant_ids": expected_uuids},
                }
            )

        def _predict(corpus_id: str) -> dict:
            return {"retrieved_ids": result_cache[corpus_id]}

        df = pd.DataFrame(rows)
        eval_result = mlflow.genai.evaluate(
            data=df,
            predict_fn=_predict,
            scorers=[retrieval_scorer],
        )

        run_id = eval_result.run_id
        retrieval_metrics = _aggregate_metrics(eval_result)

        return upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            retrieval_metrics=retrieval_metrics,
        )

    async def _run_all_predictions(
        self,
        examples: list[RetrievalCorpusExample],
        example_nodes: dict[str, tuple[KBNode, list[KBNode], dict[str, UUID]]],
    ) -> dict[str, list[str]]:
        cache: dict[str, list[str]] = {}
        for ex in examples:
            query_node, candidate_kb_nodes, _ = example_nodes[ex.corpus_id]
            await self._seed_candidates(candidate_kb_nodes)
            try:
                query = f"{query_node.title} {query_node.description}"
                results = await self._vs.search(query, top_k=5)
                cache[ex.corpus_id] = [r.node_id for r in results]
            finally:
                await self._teardown_candidates(candidate_kb_nodes)
        return cache

    async def _seed_candidates(self, nodes: list[KBNode]) -> None:
        async def _upsert(node: KBNode) -> None:
            metadata = {"node_type": node.type.value, "confidence": node.confidence}
            with contextlib.suppress(Exception):
                await self._vs.upsert(str(node.id), text=f"{node.title} {node.description}", metadata=metadata)

        await asyncio.gather(*(_upsert(node) for node in nodes))

    async def _teardown_candidates(self, nodes: list[KBNode]) -> None:
        async def _delete(node: KBNode) -> None:
            with contextlib.suppress(Exception):
                await self._vs.delete(str(node.id))

        await asyncio.gather(*(_delete(node) for node in nodes))


def _aggregate_metrics(eval_result) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("recall_at_5", "precision_at_5"):
        v = eval_result.metrics.get(f"{metric}/mean")
        if v is not None:
            result[metric] = float(v)
    return result
