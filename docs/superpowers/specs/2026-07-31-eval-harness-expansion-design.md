# Eval Harness Expansion: Splitting Retrieval into Six Harnesses — Design Spec

**Date:** 2026-07-31
**Status:** Design complete — not yet implemented

## Motivation

The eval suite (`src/seshat/eval/`) has five harnesses today: `identification`, `resolution`, `retrieval`, `grounding`, `grouping`. The `retrieval` harness calls `SearchEngine.search()` end-to-end — semantic/keyword/hybrid dispatch, multi-query fan-out, and keyword extraction are all exercised together, with no way to isolate which sub-component is responsible for a regression. Three retrieval sub-components have no eval coverage at all: the sparse (full-text) leg in isolation, the reranker (`AbstractReranker`, called by `NodeRetriever` *after* `SearchEngine.search()`, not inside it), and, indirectly, the composite pipeline that combines search + rerank the way production actually runs it.

This spec splits today's `retrieval` harness into six:

| Harness | Scope |
|---|---|
| `vector_search` | Pure dense embedding similarity — renamed from today's `retrieval` |
| `sparse_search` | Pure full-text (`ts_rank_cd`) similarity — new, symmetric to `vector_search` |
| `retrieval` | Composite: full `NodeRetriever.retrieve()` — search (however configured) → rerank |
| `keyword_extraction` | `KeywordAgent.extract()` in isolation |
| `multi_query` | `MultiQueryAgent.generate()` in isolation |
| `reranker` | `AbstractReranker.rerank()` in isolation |

All six follow the two existing structural patterns already in the codebase:
- **Mid-weight** (`vector_search`, `sparse_search`, `retrieval`): seed a vector store (and, for `retrieval`, a KB store too), run a real retrieval call, score ranked output. Same shape as today's `retrieval` harness.
- **Shallow** (`keyword_extraction`, `multi_query`, `reranker`): call one component directly, no vector store, no KB store. Same shape as `grouping` and `grounding`.

## 1. `vector_search` (renamed from today's `retrieval`)

### What it exercises

`SearchEngine.search(query, mode=SearchMode.SEMANTIC, ...)`, built with a **minimal `RAGConfig`**: `keyword_extraction_llm=None`, `multi_query.llm=None`, `reranker=None`. With these unset, `SearchEngine._semantic_search` finds no multi-query variants and calls `vector_store.search_dense()` directly — no LLM calls at all. Embedding tokens are still tracked: `get_vector_store()`'s factory already wraps the raw embeddings client in `TrackingEmbeddings`, which calls `get_run_tracker().tracker.add(embedding_input_tokens=...)` on every `aembed_query`/`aembed_documents` call, and the harness wraps its prediction loop in `@track_eval_usage("vector_search")` exactly as today's runner does — so this path is fully tracked despite making no LLM calls.

Going through `SearchEngine` (rather than calling `vector_store.search_dense()` directly) keeps this harness on the same dispatch code path production uses for `SearchMode.SEMANTIC`, and gets tracking for free instead of having to wire it independently.

### Corpus

New directory `data/eval/corpora/vector_search/`, 15 cases **copied** (not shared, not symlinked) from the existing retrieval corpus — copying means this harness's corpus can grow independently without touching the composite harness's corpus:

- `same_type_001, 002, 003, 004` — all four concept types (decision, risk, action_item, open_question)
- `cross_type_001, 002, 004, 005, 007` — five cross-type pairs
- `negative_001` (thematically adjacent), `negative_008` (large Kubernetes pool, keyword-strong)
- `realistic_001` (flyway, keyword-absent), `realistic_005` (pagerduty, keyword-absent), `realistic_011` (terraform, keyword-strong-but-semantic-agrees), `realistic_012` (slo_breach_mttr, keyword-strong-but-semantic-agrees)

**Explicitly excluded:** `realistic_015/016/017_*_sparse.yaml` (tagged `sparse_saves: true`) — these fixtures are constructed so that "a semantic-only retriever may miss it; keyword signal is the path that saves it in a hybrid retriever" (verbatim from `realistic_015`'s own description). Including them in a vector-only harness would bake in unwinnable cases and misattribute a by-design miss as a regression.

Corpus model: reuses `RetrievalCorpusExample`/`RetrievalCorpusNode` from `eval/models.py` unchanged — no new Pydantic model needed.

### Scoring

Uses the shared `recall_at_5`/`precision_at_5`/`mrr_at_5` scorer, extracted to `eval/rank_metrics.py` (see "Shared Scorers" below) — same scoring logic as today, just relocated so `sparse_search`, `retrieval`, and `reranker` can import it too instead of each owning a copy.

### Threshold calibration

`RetrievalMetaScorer` and `EvalConfig.retrieval_score_thresholds` move here (renamed `vector_search_score_thresholds`) and calibrate against `vector_search`'s corpus — they sweep the raw dense-leg cosine similarity threshold, which only this harness has raw scores for (see §3 below on why the composite harness cannot support this).

## 2. `sparse_search` (new — symmetric to `vector_search`)

### What it exercises

`SearchEngine.search(query, mode=SearchMode.KEYWORD, ...)`, built with `keyword_extraction_llm=None`. Confirmed via existing code and an existing unit test (`test_no_keyword_llm_passes_query_directly`, `tests/unit/app/pipeline/extraction/test_search_engine.py:108`) that this is already fully supported — `SearchEngine._extract_keywords` returns the raw query text when no keyword agent is configured, and `_keyword_search` passes it straight to `vector_store.search_sparse()`. No `RAGConfig` validator requires `keyword_extraction_llm` to be set for `KEYWORD` mode. **No production code change is needed** — this harness only needed confirming, not building.

This isolates the full-text ranking mechanism (`ts_rank_cd` over the generated `ts_content` tsvector column) from keyword-extraction quality, exactly the way `vector_search` isolates dense-embedding ranking from multi-query fan-out. `keyword_extraction` (§4) covers extraction quality in isolation; the composite `retrieval` harness covers the two chained together, as production actually runs them when `search_mode` is `KEYWORD` or `HYBRID`.

### Corpus

New directory `data/eval/corpora/sparse_search/`, 15 cases copied from the existing retrieval corpus, deliberately different selection criteria from `vector_search`'s — chosen to match what `ts_rank_cd` is actually good and bad at:

- `realistic_015, 016, 017` (all 3 `sparse_saves: true` cases) — this leg's signature win: cases explicitly constructed so a semantic-only retriever misses them but lexical overlap saves them.
- `realistic_011, 012, 013, 014` (`keyword_signal: strong` positives, one per concept type) — lexical match should carry these.
- `negative_008, 009, 010` (`keyword_signal: strong` negatives, large pools) — confirms shared vocabulary alone does not produce false positives; this is where a naive lexical scorer is most likely to fail.
- `negative_001, 002` (`keyword_signal: absent` negatives) and `same_type_002, 003` + `cross_type_003` (`keyword_signal: absent` positives) — baseline sanity that the leg is not purely noise on cases with no lexical signal to exploit.

**Known risk, flagged rather than resolved here:** this corpus is likely to be *harder* for `sparse_search` than the analogous 15-case corpus is for `vector_search` — `ts_rank_cd` has no semantic understanding, so the `keyword_signal: absent` cases (5 of 15) are close to a coin flip for this leg by construction, unlike `vector_search`'s corpus where semantic-absent cases are still the harness's bread and butter. Expect a materially lower baseline recall@5 than `vector_search`'s; this should inform `thresholds.py`'s `SPARSE_SEARCH_RECALL_AT_5` (likely lower than `VECTOR_SEARCH_RECALL_AT_5`) once real numbers exist, and the corpus mix itself may need rebalancing after a first run — deferred to implementation, not decided here.

### Scoring

Same shared `eval/rank_metrics.py` scorer as `vector_search` — scores are `ts_rank_cd` values rather than cosine similarities, but the recall/precision/MRR computation is score-scale-agnostic (only rank order and a threshold matter), so no scorer code is needed beyond importing it.

### Threshold calibration

`RetrievalMetaScorer`-equivalent sweep needed here too, since `ts_rank_cd` has its own score scale distinct from cosine similarity (this is already anticipated by `EvalConfig`'s existing comment: *"Each mode has its own score scale (cosine similarity for SEMANTIC, ts_rank_cd for KEYWORD, RRF for HYBRID), so thresholds must be calibrated independently"*). New `sparse_search_score_thresholds` config field, parallel to `vector_search_score_thresholds`; the meta-scorer logic itself is mode-agnostic (already parameterized over `SearchMode` via `EvalConfig.retrieval_score_thresholds.get(self._search_mode, ...)`), so this is a config/wiring addition, not new sweep logic.

## 3. `retrieval` (composite — repurposed)

### What it exercises

The full production retrieval path as `ExtractionOrchestrator` actually assembles it: `NodeRetriever.retrieve(query_kb_node, node_filter=NodeFilter(node_type=None))`, using `seshat_config.rag` exactly as configured (real `search_mode`, real `keyword_extraction_llm`, real `multi_query`, real `reranker` — whatever the running config specifies). Internally this calls `SearchEngine.search()` (dispatching through keyword extraction / multi-query fan-out / hybrid RRF per config) and then, if a reranker is configured, `AbstractReranker.rerank()`.

**Seeding:** each candidate node is written via `NodeRepository.write_node()` — one call that keeps a dedicated eval KB schema and a dedicated eval vector collection in sync (mirrors today's `_EVAL_COLLECTION` isolation pattern, extended with a dedicated eval KB schema, e.g. `seshat_retrieval_eval`). No relationships are seeded, so `NodeRetriever._expand_with_neighbours`'s graph traversal naturally finds nothing extra for any candidate — this keeps the harness from needing to special-case `traversal_max_depth` or `traversal_rel_types`, without touching `NodeRetriever` itself.

**Teardown:** `NodeRepository.delete_node()` per candidate, symmetric with the seed step, keeping both stores clean between examples.

This means the harness re-uses `NodeRetriever` verbatim rather than reimplementing its search→rerank sequencing — any future change to `NodeRetriever`'s internals (budget interactions, conditional reranking, cap logic) is automatically covered without the harness drifting out of sync.

### Why there is no threshold parameter here

`NodeRetriever.retrieve()` returns `list[KBNode]` — no similarity scores. Today's runner filters `retrieved_ids` by a calibrated threshold before scoring; that machinery has nothing to operate on here. This is intentional, not a gap: `NodeRetriever`'s own `min_similarity_score` (passed as `score_threshold` into `SearchEngine.search()`) and the reranker's `top_n` **are** the filtering behavior under test. The composite harness scores the top-5 of whatever ranked list production code hands back, full stop — it validates the assembled pipeline end-to-end, it does not calibrate any individual knob. Calibration of the dense-leg and sparse-leg thresholds stays with `vector_search` (§1) and `sparse_search` (§2) respectively.

### Corpus

Unchanged: the existing 40-file `data/eval/corpora/retrieval/`, including the 3 `sparse_saves: true` cases — this is exactly the harness where a fully hybrid-configured, reranked pipeline is expected to win them.

### Scoring

Uses the shared `eval/rank_metrics.py` scorer (see "Shared Scorers" below) — same recall/precision/mrr@5 computation as today, relocated.

## 4. `keyword_extraction` (shallow)

### What it exercises

`KeywordAgent.extract(query)` directly. No vector store, no search.

### Corpus

New Pydantic model in `eval/models.py`:

```python
class KeywordExtractionCorpusExample(BaseModel):
    corpus_id: str
    query: str  # combined title + description text, as SearchEngine would pass it
    expected_keywords: list[str]  # gold discriminating terms; case-insensitive substring match
    banned_terms: list[str] = Field(default_factory=list)  # generic terms that should not leak through
    tags: dict[str, Any] = Field(default_factory=dict)
```

New corpus directory `data/eval/corpora/keyword_extraction/`. Cases mirror the kind of query nodes already in the retrieval corpus (e.g. the Flyway decision → expected keywords `["Flyway", "migration"]`, banned `["decision", "team"]`), authored fresh rather than mechanically derived — the point is discriminating-term judgment, not query-text reuse.

### Scoring

```python
@mlflow.genai.scorer
def scorer(inputs, outputs, expectations) -> list[Feedback]:
    """keyword.recall (gated): fraction of gold keywords found (case-insensitive substring) in the
    extracted keyword string. keyword.banned_leak_rate (logged): fraction of banned terms that leaked through."""
```

`keyword.recall` gated (mirrors `grouping.group_hit_rate`'s partial-credit pattern); `keyword.banned_leak_rate` logged only.

**Why not `grouping`'s `exact_match` too:** grouping's gold set (`expected_groups`) is an *exhaustive* correct partition — any deviation, extra or missing, is a real error, so both a strict (`exact_match`) and partial-credit (`group_hit_rate`) view of the same ground truth make sense. `expected_keywords` here is a "must-include checklist," not an exhaustive definition of the only acceptable output: an extractor that returns `["Flyway", "migration", "schema"]` against a gold set of `["Flyway", "migration"]` is not wrong, and may be an improvement. An `exact_match`-style metric would penalize legitimately useful extra discriminating terms, so it is deliberately omitted — `keyword.recall` (checklist coverage) + `keyword.banned_leak_rate` (explicit contamination check) is the complete metric set, not a partial one.

## 5. `multi_query` (shallow)

### What it exercises

`MultiQueryAgent.generate(query)` directly, then embeds `[query, *variants]` via the plain (non-`TrackingEmbeddings`-required, but tracked the same way for consistency) embeddings client — no vector store, no LLM judge.

### Corpus

```python
class MultiQueryCorpusExample(BaseModel):
    corpus_id: str
    query: str
    tags: dict[str, Any] = Field(default_factory=dict)
```

No `expected_*` field — scoring is reference-free (there is no single "correct" paraphrase to match against). New corpus directory `data/eval/corpora/multi_query/`.

### Scoring

Two metrics, both **logged only, not gated**:

- `multi_query.fidelity` — mean cosine similarity of each variant's embedding to the original query's embedding. Should stay reasonably high (same intent, different phrasing).
- `multi_query.diversity` — mean pairwise dissimilarity (1 − cosine similarity) among variant embeddings. Should be non-trivial — otherwise multi-query fan-out adds cost without adding recall.

**Why not gated:** the embedding step itself is deterministic given fixed text, but the LLM call generating variants is not — repeated runs against an unchanged prediction cache are stable, but a cache-busting run (new prompt, new model) will legitimately produce different variants each time, and there is no ground-truth "correct" variant to gate against, unlike `keyword_extraction`'s discriminating-term gold set. Hard thresholds on a reference-free signal would produce a flaky release gate; logging keeps the signal visible in MLflow without blocking releases on LLM sampling variance.

## 6. `reranker` (shallow)

### What it exercises

`AbstractReranker.rerank(query_text, candidate_kb_nodes)` directly, built via `app/pipeline/bootstrap.py::_get_reranker(seshat_config)` — the same factory production uses (resolves the API key through `get_secrets_resolver`, dispatches to `CohereReranker`/`VoyageReranker` via `reranker_factory`). No search call, no vector store: candidates are constructed as `KBNode` objects straight from the corpus (reusing the `_to_kb_node` helper already used by `vector_search`/`retrieval`/`sparse_search`), and handed to `rerank()` in a **shuffled** order — never the corpus's original listing — so the test exercises reordering rather than trivially preserving whatever order the fixture happens to list candidates in.

**Shuffle determinism:** each corpus example's candidate list is shuffled with `random.Random(seed)`, where `seed` is derived from the corpus_id (e.g. `int(fingerprint(corpus_id), 16)`, reusing `core/utils/hashing.fingerprint`) — deterministic and reproducible across runs (same corpus_id always shuffles the same way, so a diff in `recall_at_5` between two runs reflects a real reranker behavior change, not shuffle noise), while still decorrelating input order from relevance order (unlike a fixed reverse-order, which is itself a special-case ordering a reranker could coincidentally do well or badly against).

**Config fallback, mirroring the `grounding` harness's existing pattern** (`cli/_eval_support.py:59-61`, which injects a default `GroundingLLMConfig()` with a warning when `seshat_config.extraction.grounding is None`): if `seshat_config.rag.reranker is None`, `bootstrap_eval("reranker")` injects a default `RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5")` — the exact example already named in `RerankerConfig`'s own field docstring — and logs a warning, rather than failing hard. This keeps `seshat eval harness reranker` runnable out of the box while still requiring a Cohere API key to be resolvable via secrets (same trust boundary as every other real-provider-calling harness — grounding, identification, etc. — none of which mock the provider away).

### Corpus

Reuses `RetrievalCorpusExample`/`RetrievalCorpusNode` unchanged (already harness-agnostic: query node + candidate pool + expected relevant IDs is exactly the shape reranker eval needs). New corpus directory `data/eval/corpora/reranker/`, 15 cases copied from the existing retrieval corpus (no `sparse_saves` exclusion needed here, since reranking is scored on relevance ranking, not on the sparse leg) — same selection principle as `vector_search`'s corpus (balanced across concept types, cross-type pairs, and negatives), file list to be finalized during implementation (see Open Questions).

### Scoring

Uses the shared `recall_at_5`/`precision_at_5`/`mrr_at_5` scorer from `eval/rank_metrics.py` (see "Shared Scorers" below) — reranker eval is "given this shuffled pool, does reordering surface the right nodes in the top 5," the same scoring question as `vector_search`/`sparse_search`/`retrieval`, just with a deliberately adversarial input order instead of a natural one.

### Threshold calibration

None. `AbstractReranker.rerank()` returns a re-ordered `list[KBNode]`, no relevance scores — same situation as the composite `retrieval` harness (§3) and for the same reason: `top_n` (if configured) is the only "filtering" knob, and it's part of what's under test, not something this harness calibrates. Scores top-5 of whatever order the reranker returns.

## Shared Scorers

`recall_at_5`/`precision_at_5`/`mrr_at_5` is now needed verbatim by four harnesses (`vector_search`, `sparse_search`, `retrieval`, `reranker`), all scoring the same shape of output (a ranked list of node IDs against an expected-relevant-IDs set). Extract it once to a new **`eval/rank_metrics.py`** — a flat module directly under `eval/`, matching the existing convention for suite-wide (not harness-specific) code: `eval/cache.py`, `eval/gate.py`, `eval/models.py`, `eval/thresholds.py`, and `eval/corpus_tags.py` are all flat siblings of the per-harness packages, not nested under any one harness. `rank_metrics.py` holds `TOP_K = 5` and the `@mlflow.genai.scorer` function, moved from today's `eval/retrieval/scorers.py` unchanged. All four harnesses' `scorers.py` becomes a one-line re-export (`from seshat.eval.rank_metrics import scorer, TOP_K`) so each harness's directory still has a `scorers.py` for structural consistency with `identification`/`resolution`/`grouping`/`grounding`/`keyword_extraction`/`multi_query`, but the logic itself lives in one place.

`keyword_extraction` and `multi_query`'s scorers are genuinely harness-specific (term-overlap recall; embedding fidelity/diversity) and stay owned by their own `scorers.py`, same as `grouping`/`grounding` today.

## CLI Changes

`seshat eval harness <name>` and `seshat eval clear-cache <name>` (`cli/app.py`, `cli/_eval_support.py`) already dispatch generically over `HARNESS_TYPES` — adding five names to that list plus five `case` arms in `cli/app.py::_run_single_harness`'s `match` (already itemized under "Plumbing Changes" above) is sufficient for `harness` and `clear-cache` to pick up all five new harnesses with no other code change, since `eval_cmd`'s `--all` path, per-harness `try/except` loop, and `_clear_cache` are all already generic over the list.

**`seshat eval calibrate`** needs a second, harness-specific change beyond the mechanical list update: `CALIBRATION_TYPES` (`cli/_eval_support.py:30`) currently reads `["retrieval", "identification"]`, and `cli/app.py`'s `calibrate_cmd` match-statement (`cli/app.py:166-178`) has a `case "retrieval":` arm importing `eval.calibration.retrieval_entrypoint`. Since threshold calibration moves to `vector_search` (§1) and gains a new `sparse_search` counterpart (§2):

- `CALIBRATION_TYPES` becomes `["vector_search", "sparse_search", "identification"]` — the composite `retrieval` harness has no threshold to calibrate (§3), so it is **not** added here, and the old `"retrieval"` entry is removed rather than kept as an alias.
- `eval/calibration/retrieval_entrypoint.py` + `retrieval_meta_scorer.py` are renamed `vector_search_entrypoint.py` + `vector_search_meta_scorer.py` (per the Directory Layout section below), and a new `sparse_search_entrypoint.py` + `sparse_search_meta_scorer.py` pair is added, parallel in structure — same `RetrievalMetaScorer`-shaped class (renamed `SparseSearchMetaScorer` or kept generic — naming TBD in implementation), built with `search_mode` fixed to `KEYWORD` instead of reading it from `seshat_config.rag.search_mode`, since the whole point of this calibration run is to sweep the sparse leg specifically regardless of what mode is currently configured.
- `calibrate_cmd`'s `match` gains `case "vector_search":` and `case "sparse_search":` arms (replacing the old `case "retrieval":`), each importing its respective `run`.
- `--pc-curve` remains identification-only (unaffected — it's gated behind `component == "identification"`'s branch already).

**Existing calibration behavior preserved:** `bootstrap_eval(f"{component}-calibration")` in `_eval_support.py` is component-name-agnostic (just formats a job/run name string), so no change needed there beyond the component names themselves flowing through correctly.

## Gate Mechanism

The gate itself is already generic where it can be (`GateResult.passed`/`harness_passed` iterate a list of `dict[str, MetricEntry] | None` blocks with no per-harness branching), but three call sites hardcode the current five block names and need updating:

1. **`upsert_gate` and the five `*_entries` converters** (`gate.py`) — already itemized under "Plumbing Changes": five new optional kwargs on `upsert_gate`, five new `*_entries` functions mirroring the existing `retrieval_entries`/`grouping_entries` shape.
2. **`GateResult`'s five `Optional[dict]` fields** (`eval/models.py`) — already itemized: five new fields. `GateResult.passed`'s `all_metrics` list and `harness_passed`'s `getattr(self, f"{harness}_metrics")` lookup both already iterate generically, so **no logic change** is needed inside `GateResult` beyond adding the five fields themselves — the existing AND-of-all-gated-blocks semantics extends automatically to cover the new blocks the moment they're populated.
3. **`app/platform/api/app.py::_check_eval_gate`** — reads the gate file as raw JSON (`json.loads(gate_path.read_text())`) and checks only the top-level `"passed"` boolean, never touching individual harness blocks. Since `passed` is computed the same generic way regardless of how many blocks exist, **this call site needs no change at all** — a `vector_search_metrics` or `reranker_metrics` block dragging `passed` to `false` is already handled by existing code, exactly as `grouping_metrics` failing would be today.

**Net effect:** the gate *file schema* grows (five new optional blocks) and gate.py's converters grow (five new functions, mechanical), but the gate *mechanism* — computing `passed`, blocking API startup on it, carrying over untouched blocks on a partial `upsert_gate` call — needs zero behavioral changes, because it was already written generically over "some set of named metric blocks." This is worth stating explicitly since it's easy to assume a bigger gate rewrite is needed; it isn't.

**One open item:** should the release gate (`_check_eval_gate`) require the five new harnesses to be present/passing before the API will start, the same as the existing five? Since `GateResult.passed` is a strict AND of every *present* block (a `None` block is not counted, per the existing docstring: *"A `None` block means the pass was not run and is not a failure"*), a fresh `data/eval_gate.json` that has never run the new harnesses simply omits those blocks and `passed` is computed from the remaining five, unaffected. This means **no migration is forced** — teams adopt the new harnesses at their own pace by choosing to run them (or not) via `EvalConfig.run_<harness>` flags, and the gate only starts counting a harness once it has actually been run at least once. This is the same rollout behavior the suite already has today for any harness whose flag is toggled off.

## Plumbing Changes

All mechanical, following the existing per-harness pattern:

- **`EvalConfig`** (`core/config/eval_settings.py`): add `run_vector_search`, `run_sparse_search`, `run_keyword_extraction`, `run_multi_query`, `run_reranker` bool flags (default `True`); add `_vector_search_subdir`, `_sparse_search_subdir`, `_keyword_extraction_subdir`, `_multi_query_subdir`, `_reranker_subdir` ClassVars and corresponding `*_corpus_dir`/`*_cache_dir` properties; composite `retrieval` keeps its existing corpus dir property unaffected; update `enabled_harnesses`, `_validate_corpus_dirs`, `_create_cache_dirs` to include the five new harnesses. Rename `retrieval_score_thresholds` → `vector_search_score_thresholds` (env var `EVAL__VECTOR_SEARCH_SCORE_THRESHOLDS__*`); add `sparse_search_score_thresholds` (env var `EVAL__SPARSE_SEARCH_SCORE_THRESHOLDS__*`).
- **`HARNESS_TYPES`** (`cli/_eval_support.py`) gains `"vector_search"`, `"sparse_search"`, `"keyword_extraction"`, `"multi_query"`, `"reranker"`.
- **`cli/app.py::_run_single_harness`**: five new `case` arms importing each new `entrypoint.run`.
- **`CALIBRATION_TYPES`** (`cli/_eval_support.py`) becomes `["vector_search", "sparse_search", "identification"]` (drops `"retrieval"` — see "CLI Changes" above for why); **`cli/app.py::calibrate_cmd`**'s `match` gains `case "vector_search":`/`case "sparse_search":` arms replacing the old `case "retrieval":`.
- **`GateResult`** (`eval/models.py`): add `vector_search_metrics`, `sparse_search_metrics`, `keyword_extraction_metrics`, `multi_query_metrics`, `reranker_metrics` blocks (`dict[str, MetricEntry] | None`); `retrieval_metrics` is retained unchanged (now scoped to the composite pass); `passed` / `harness_passed` iterate the same way since both are already generic over "all present blocks."
- **`thresholds.py`**: `VECTOR_SEARCH_RECALL_AT_5` / `VECTOR_SEARCH_MRR_AT_5` (renamed from today's `RETRIEVAL_*`); `SPARSE_SEARCH_RECALL_AT_5` / `SPARSE_SEARCH_MRR_AT_5` (new, likely lower bar — see §2's known-risk note); `RETRIEVAL_RECALL_AT_5` / `RETRIEVAL_MRR_AT_5` retained for the composite pass — likely calibrated to a *different* (probably higher, since reranking is in the loop) bar once real data exists, but starting at the same values as today is the safe default; `KEYWORD_EXTRACTION_RECALL`; `RERANKER_RECALL_AT_5` / `RERANKER_MRR_AT_5`. No threshold constants for `multi_query` (nothing gated).
- **`gate.py`**: `vector_search_entries`, `sparse_search_entries`, `keyword_extraction_entries`, `reranker_entries` follow the existing `retrieval_entries`/`grouping_entries` shape exactly; `multi_query_entries` is the same shape but its `gate_judge` always returns `None` (nothing gated, everything logged) — mirrors how `grouping_entries` already treats `exact_match` as logged-only within an otherwise-gated block.
- **`upsert_gate`**: five new optional kwargs, following the existing `carry-over-if-not-supplied` pattern for all six (now ten) blocks.

## Directory Layout (new)

```
src/seshat/eval/
├── rank_metrics.py          # new: shared TOP_K + recall/precision/mrr@5 scorer (see Shared Scorers)
├── vector_search/           # renamed from retrieval/, minimal-config SearchEngine.search(mode=SEMANTIC)
│   ├── __init__.py
│   ├── corpus_loader.py     # moved from retrieval/, unchanged
│   ├── entrypoint.py
│   ├── runner.py
│   └── scorers.py           # re-exports from rank_metrics.py
├── sparse_search/           # new, minimal-config SearchEngine.search(mode=KEYWORD)
│   ├── __init__.py
│   ├── corpus_loader.py     # reuses RetrievalCorpusExample/build_kb_nodes
│   ├── entrypoint.py
│   ├── runner.py
│   └── scorers.py           # re-exports from rank_metrics.py
├── retrieval/               # repurposed: composite NodeRetriever.retrieve()
│   ├── __init__.py
│   ├── corpus_loader.py     # unchanged
│   ├── entrypoint.py        # rewritten: builds NodeRepository, seeds via write_node
│   ├── runner.py            # rewritten: no threshold param, no vector_store arg (uses node_repo)
│   └── scorers.py           # re-exports from rank_metrics.py
├── keyword_extraction/      # new, shallow
│   ├── __init__.py
│   ├── corpus_loader.py
│   ├── entrypoint.py
│   ├── runner.py
│   └── scorers.py           # harness-specific (term-overlap recall), not shared
├── multi_query/             # new, shallow
│   ├── __init__.py
│   ├── corpus_loader.py
│   ├── entrypoint.py
│   ├── runner.py
│   └── scorers.py           # harness-specific (fidelity/diversity), not shared
├── reranker/                # new, shallow (reuses retrieval's corpus model)
│   ├── __init__.py
│   ├── corpus_loader.py     # thin wrapper around retrieval's load_corpus + build_kb_nodes
│   ├── entrypoint.py
│   ├── runner.py
│   └── scorers.py           # re-exports from rank_metrics.py
└── calibration/
    ├── vector_search_entrypoint.py    # renamed from retrieval_entrypoint.py
    ├── vector_search_meta_scorer.py   # renamed from retrieval_meta_scorer.py, points at vector_search corpus
    ├── sparse_search_entrypoint.py    # new, parallel structure
    └── sparse_search_meta_scorer.py   # new, parallel sweep over ts_rank_cd thresholds

data/eval/corpora/
├── vector_search/         # new, 15 copied cases
├── sparse_search/         # new, 15 copied cases (different selection — see §2)
├── retrieval/             # unchanged, 40 cases
├── keyword_extraction/    # new
├── multi_query/           # new
└── reranker/              # new, curated subset copied from retrieval/
```

## Implementation Approach

No separate `docs/superpowers/plans/*.md` document is written for this work — a prose task plan for a six-harness spec tends to drift out of sync with the code as implementation proceeds, and this spec (plus the existing harnesses as concrete templates) is detailed enough to implement directly from. Work proceeds test-first, straight from this spec, in three sequential tiers — each tier lands as its own PR and must leave the suite in a working, gate-passing state before the next tier starts:

- **Tier 1** — `eval/rank_metrics.py` extraction (§"Shared Scorers") + `vector_search` rename/split (§1) + `sparse_search` (§2, new) + the `calibrate` CLI rewiring (§"CLI Changes"). Bundled because all four share the same minimal-config-`SearchEngine` + corpus-copy + threshold-calibration pattern, and `sparse_search` is easiest to get right immediately after `vector_search` while that pattern is fresh.
- **Tier 2** — `keyword_extraction` (§4), `multi_query` (§5), `reranker` (§6). Three mutually independent shallow harnesses; may be implemented in any order or in parallel, but each still lands as part of one PR since none touches the others' files beyond the same handful of shared-config additions (`EvalConfig`, `HARNESS_TYPES`, `GateResult`, `thresholds.py`, `gate.py`).
- **Tier 3** — composite `retrieval` rewrite (§3) alone. Largest and riskiest unit (dual-store `NodeRepository` seeding, dropping the threshold param, most different in shape from every other harness) — isolated last so a problem here does not block the other five harnesses from shipping.

**Templates to mirror, not reinvent:** `eval/retrieval/` (today's mid-weight pattern: corpus_loader/entrypoint/runner/scorers, vector-store seed+teardown) for `vector_search`/`sparse_search`/`retrieval`; `eval/grouping/` and `eval/grounding/` (shallow pattern: direct agent call, no vector store) for `keyword_extraction`/`multi_query`/`reranker`. Each new harness is a close structural copy of its template with the specifics substituted in per this spec's §-by-§ description, not a from-scratch design.

**Within each tier:** write the failing test against the spec's stated behavior (e.g. "`SearchEngine.search(mode=KEYWORD)` with `keyword_extraction_llm=None` passes the raw query to `search_sparse`" for `sparse_search`, verified today by the existing `test_no_keyword_llm_passes_query_directly` unit test — new harness-level tests follow the same shape), then implement to green, following the existing per-harness test layout under `tests/`.

## Open Questions / Deferred Decisions (resolved during implementation, not this spec)

1. **`sparse_search`'s corpus difficulty and threshold** — flagged in §2 as likely harder than `vector_search`'s equivalent corpus by construction (no semantic understanding in `ts_rank_cd`). The initial 15-case mix and `SPARSE_SEARCH_RECALL_AT_5` are starting points, not calibrated values — expect to revisit both after a first real run.
2. **Reranker corpus subset size and file list** — 15-case target set (§6), specific files to be finalized during implementation, following the same reuse-with-explicit-list approach used for `vector_search`.
3. **Keyword_extraction and multi_query corpus case counts and specific content** — this spec defines the corpus *model* and *scoring*, not the actual fixture authoring, which happens during implementation.
4. **`SparseSearchMetaScorer` class naming** — §"CLI Changes" leaves open whether the sparse-leg calibration class is a distinct `SparseSearchMetaScorer` or a generalized/parameterized version of the existing `RetrievalMetaScorer` (renamed `VectorSearchMetaScorer`); both sweep the same shape of threshold curve over a different score source, so a shared base or a `search_mode` constructor arg are both plausible — decide during Tier 1 once both entrypoints are written side by side.
