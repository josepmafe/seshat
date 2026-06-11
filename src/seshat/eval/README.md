# `seshat.eval` — Evaluation Harnesses

Five eval harnesses that measure quality across the pipeline:
**identification**, **resolution**, **retrieval**, **grouping**, and **verification**.
All five use [MLflow Evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate/) as the
evaluation framework and write their results to a shared gate file.

## Installation

The eval package requires optional dependencies.  Install them with:

```bash
uv sync --group eval
```

## Architecture

```
eval/
├── identification/       # ExtractionOrchestrator → KBNodes
│   ├── corpus_loader.py    reads YAML fixtures, builds IdentificationCorpusExample
│   ├── matcher.py          greedy bipartite span-overlap matcher
│   ├── scorers.py          precision/recall + field-accuracy feedback
│   └── runner.py           IdentificationEvalRunner
├── resolution/           # ExtractionOrchestrator → relationships
│   ├── corpus_loader.py    reads YAML fixtures, builds KBNodes from slugs
│   ├── scorers.py          exact-triple precision/recall feedback
│   └── runner.py           ResolutionEvalRunner
├── retrieval/            # AbstractVectorStore → top-k search
│   ├── corpus_loader.py    reads YAML fixtures, builds KBNodes
│   ├── scorers.py          recall@5 / precision@5 feedback
│   └── runner.py           RetrievalEvalRunner
├── grouping/             # GroupingAgent → grouped extraction quality
│   ├── corpus_loader.py    reads YAML fixtures, builds GroupingCorpusExample
│   ├── scorers.py          exact_match + group_hit_rate feedback
│   └── runner.py           GroupingEvalRunner
├── verification/         # VerificationAgent → quote grounding quality
│   ├── corpus_loader.py    reads YAML fixtures, builds VerificationCorpusExample
│   ├── scorers.py          confusion-matrix feedback (TP/FP/FN/TN)
│   └── runner.py           VerificationEvalRunner
├── calibration/
│   ├── identification_meta_scorer.py   IdentificationMetaScorer
│   ├── retrieval_meta_scorer.py        RetrievalMetaScorer
│   └── models.py                       SweepPoint, SweepResult, TypeMetrics, etc.
├── cache.py              # read_or_run helper
├── common.py             # shared helpers (not public API)
├── gate.py               # read/write/upsert gate file
├── models.py             # GateResult, corpus Pydantic models
└── thresholds.py         # pass/fail thresholds (in code, not config)
```

---

## The Gate

Every runner produces a `GateResult` and writes it to the `gate_path` configured in
`EvalConfig` via `upsert_gate`.  `upsert_gate` is additive: running one harness
carries over the other harnesses' metric blocks so you can run them independently
without zeroing out prior results.

`GateResult.passed` computes the overall pass/fail verdict by comparing all present
metric blocks against the thresholds in `thresholds.py`.  A block that is `None`
(not yet run) is skipped — it does not fail the gate.

Thresholds are in code, not config, so that lowering them requires a reviewable code
change.

---

## MLflow trace design

### Why `_predict` is structured the way it is

`mlflow.genai.evaluate` calls `predict_fn` synchronously, passing the `inputs`
columns from the evaluation DataFrame as keyword arguments.  The parameter names
of `_predict` must exactly match the keys in the `inputs` dict — MLflow unpacks by
name.

The actual LLM work does **not** happen inside `_predict`.  All orchestrator calls
are async; MLflow's sync loop can't await them.  Instead, every runner
pre-populates a `result_cache` via `_run_all_predictions` before handing control to
`mlflow.genai.evaluate`.  `_predict` is then just a cache lookup.

The underscore-prefixed columns in the resolution DataFrame (`_source_nodes`,
`_kb_nodes`) are a naming convention that signals intent: they are corpus context
forwarded into the trace for observability, not inputs that drive the computation.
The computation already happened during `_run_all_predictions`.

### What goes into the trace output, and why

`_predict` returns more than the scorer strictly needs.  Alongside the
scorer-consumed field (`nodes` / `relationships`), it also includes a human-readable
comparison view:

- **Identification** — `expected_nodes` (sans `quote`) next to `nodes`.
- **Resolution** — `predicted_relations` and `expected_relations` both expressed as
  slug triples, next to the UUID-based `relationships` the scorer uses.

The reason: in the MLflow UI each fixture is one row.  A failing fixture should be
fully diagnosable from that row alone — without opening the corpus YAML or
cross-referencing another panel.  By placing expected and predicted side-by-side in
the trace output, a reviewer can see the diff immediately.

### Trace output slimming

The identification runner registers `_slim_output_nodes` as a trace post-processor
via `configure_trace_processors`.  It strips verbose fields (raw `quote_anchors`,
long `description` strings) from the span outputs and keeps only
`type / title / description / confidence`.

The goal is the same as above: debuggability.  A raw `KBNode` dump is noisy.  The
slimmed version surfaces what matters — did the model find the right type and give
it a sensible title? — without scrolling through serialised spans.

---

## Eval determinism

Results are **not fully deterministic** across runs. This has been observed empirically
across repeated eval runs on the same corpus — identical prompts and `temperature=0`
settings occasionally yield different outputs. This is consistent with the scientific
literature: frontier models exhibit residual non-determinism even at `temperature=0`,
attributed to floating-point non-associativity in parallel GPU kernels and
provider-side batching strategies (see e.g. Chen et al., 2023; OpenAI's own
documentation acknowledges this for GPT-4 class models).

Two sources of variance in this harness:

1. **LLM sampling** — all runners call live LLM endpoints with `temperature=0`. This
   is the dominant source. Variance is small (we rarely see a fixture flip category),
   but it is real.
2. **Retrieval UUID assignment** — the retrieval runner generates UUIDs via `uuid4()`
   (random per call). If a fixture's candidate nodes receive different UUIDs on
   different runs they will produce different embeddings in the vector store and can
   yield different recall@5 scores for the same fixture.

The identification and resolution runners use `uuid5` (deterministic from
`corpus_id + slug`) so their node identities are stable across runs.

**How we manage variance:**

- The corpus is kept small and focused — each fixture tests a specific, unambiguous
  case. This minimises the surface area where LLM variance can shift a result.
- Boundary cases where the correct label is genuinely ambiguous are documented in
  `grey_area` notes inside the fixture YAML. These are the cases most likely to flip
  across runs; the notes record the reasoning so a score change can be diagnosed
  quickly rather than treated as a regression.
- Use `read_or_run` caching (via `eval/cache.py`) during iterative development to pin
  predictions and isolate scorer changes from LLM variance. The cache is keyed on the
  corpus file hash and is cleared automatically at the end of a successful run.

---

## Harness 1 — Identification (`IdentificationEvalRunner`)

**What it measures:** Can the extraction pipeline find the right nodes in a meeting
transcript and populate their fields correctly?

**How it works:**

1. Loads YAML corpus fixtures from `EvalConfig.identification_corpus_dir`.
2. For each fixture, runs `ExtractionOrchestrator._run_identification` against the
   transcript (async, pre-populated into a cache before MLflow's sync loop runs).
3. Feeds the resulting `KBNode` list through the **greedy bipartite span-overlap matcher**
   in `matcher.py`.  Matching prefers `quote_anchors` spans; falls back to weighted
   title+description fuzzy similarity when no quote anchor is present.
4. The `scorer` function produces per-`ConceptType` precision/recall/F1 feedback, plus
   per-field accuracy scores (fuzzy for `assignee`, `due`, `rationale`, `context`;
   exact for `type`; set-overlap for `alternatives_considered`).
5. Gate metrics: per-type precision and recall (dotted keys like
   `action_item.precision`).  Field-level scores are logged to MLflow as observability
   signals but are **not** gate criteria.

**Corpus fixture format** — one YAML file per meeting scenario:

```yaml
transcript: |
  ...full meeting transcript...
expected:
  action_item:
    - quote: "exact substring from transcript"
      title: "Short label"
      description: "Full description"
      assignee: "Alice"          # optional extra field
      due: "2026-06-01"
  decision:
    - quote: "..."
      title: "..."
      description: "..."
      rationale: "..."
```

**Limitations:**
- Quote matching uses fuzzy partial ratio — a predicted node whose quote is a
  substring of a different sentence can still match if the score exceeds the 0.70
  threshold.
- Field accuracy scores cover only structured fields; semantic quality of `title` and
  `description` is not scored.
- Greedy matching: if two predicted nodes are both good candidates for the same
  expected node, only the higher-scoring one is counted as a true positive.
- Does not test the full ingestion pipeline (no S3, no Postgres, no deduplication).

---

## Harness 2 — Resolution (`ResolutionEvalRunner`)

**What it measures:** Can the pipeline correctly link new nodes to existing KB nodes
via `SUPERSEDES`, `AMENDS`, `CONFLICTS_WITH`, `DEPENDS_ON`, `MITIGATES`, `BLOCKS`,
or `RESOLVES` relationships?

**How it works:**

1. Loads YAML corpus fixtures from `EvalConfig.resolution_corpus_dir`.
2. Converts corpus slugs to UUIDs via `uuid5(NAMESPACE_URL, f"{corpus_id}/{slug}")` —
   deterministic across runs; the same slug always maps to the same UUID for a given
   fixture.
3. For each fixture, calls `ExtractionOrchestrator._run_resolution` with the
   `source_nodes` and a `per_source_targets` map covering all KB nodes in the fixture.
4. The `scorer` computes precision/recall over exact `(source_id, target_id, rel_type)`
   triples.

**Corpus fixture format:**

```yaml
corpus_id: decision-supersedes-001
description: "A revised decision supersedes an earlier one"
source_nodes:
  - id: new-decision
    type: decision
    title: "..."
    description: "..."
    quote: "..."
kb_nodes:
  - id: old-decision
    type: decision
    title: "..."
    description: "..."
    quote: "..."
expected_relations:
  - source: new-decision
    target: old-decision
    rel_type: SUPERSEDES
```

**Heuristic validation (pipeline-level, not eval-level):**

The pipeline applies two post-prediction filters before relationships reach the KB.
The eval scorer operates on post-validation output, so these constraints are not
visible as FPs in eval metrics — a violation is silently dropped before scoring:

- **Mutually exclusive pairs** — `(supersedes, conflicts_with)` and
  `(supersedes, amends)` cannot both hold for the same `(source, target)` pair.
  When both appear, `amends` is preferred (the weaker claim); the other is dropped.
- **Anti-symmetric relations** — `supersedes`, `blocks`, and `depends_on` cannot
  hold in both directions simultaneously (A→B and B→A). The entire pair is dropped.

**Limitations:**
- Fixtures set `per_source_targets` to the full KB nodes list for every source node.
  Real runs use a filtered candidate set from the retrieval pass — the eval therefore
  tests resolution quality assuming perfect retrieval, not end-to-end quality.
- Exact triple matching: an `AMENDS` when `SUPERSEDES` was expected scores zero, even
  if the relationship is directionally correct.
- Does not cover ambiguous cases where multiple valid relationship types exist.

---

## Harness 3 — Retrieval (`RetrievalEvalRunner`)

**What it measures:** Can the vector store surface relevant KB nodes in the top-5
results for a given query node?

**How it works:**

1. Loads YAML corpus fixtures from `EvalConfig.retrieval_corpus_dir`.
2. For each fixture, seeds the `AbstractVectorStore` collection with the candidate
   nodes, issues a `search(title + description, top_k=5)` query, then tears down
   (delete) the candidates — all within a try/finally to avoid leaving stale data.
3. `build_kb_nodes` generates UUIDs via `uuid4()` — random per call. It must be called
   **once** per fixture per run; calling it twice produces different UUIDs and breaks
   the slug→UUID mapping used to resolve `expected_relevant_ids`.
4. The `scorer` computes recall@5 and precision@5 against the `expected_relevant_ids`.

**Corpus fixture format:**

```yaml
corpus_id: retrieval-action-001
description: "Should retrieve a related action item"
tags:
  tier: same_type        # same_type | cross_type | realistic | negative
  source_type: action_item
  polarity: positive     # positive | negative
  difficulty: medium     # easy | medium | hard
query_node:
  id: new-action
  type: action_item
  title: "Deploy hotfix"
  description: "Deploy the fix to production"
  quote: "..."
candidate_nodes:
  - id: existing-action
    type: action_item
    title: "Deploy release"
    description: "Deploy the release build"
    quote: "..."
  - id: unrelated-decision
    ...
expected_relevant_ids:
  - existing-action      # empty list [] for negative-polarity fixtures
```

**File naming convention:** `<tier>_<nnn>_<description>.yaml`, matching the pattern used
by the identification and resolution corpora.  The tier prefix enables `ls`-level
grouping and makes the fixture's role legible without opening the file:

| Tier | Count | Pool composition | Polarity |
|---|---|---|---|
| `same_type` | 4 | single-type pool, same-type query→candidate pairing | positive |
| `cross_type` | 9 | single-type pool, cross-type query→candidate pairing | positive |
| `realistic` | 10 | mixed-type pool (all 4 node types), reflects production distribution | positive |
| `negative` | 7 | mixed-type pool, `expected_relevant_ids: []` | negative |

**Limitations:**
- The caller **must** pass a dedicated, empty collection.  Any pre-existing nodes in
  the collection will appear in search results and corrupt scores.
- Seed/teardown is best-effort: exceptions during upsert or delete are suppressed.  A
  failed teardown leaves nodes in the store and will skew subsequent fixtures in the
  same run.
- Evaluates embedding-based retrieval only — does not test re-ranking or any
  post-retrieval filtering.
- Query is constructed as `title + " " + description`; the production pipeline may
  use a different query strategy.
- Gate criterion is recall@5 ≥ 0.70; precision@5 is logged but not gated.
- Corpus fixtures use mixed-type candidate pools (decisions, risks, action items, open
  questions in the same pool) to simulate a realistic vector store. Single-type pools
  suppress false positives and produce misleadingly high precision scores.

---

## Harness 4 — Grouping (`GroupingEvalRunner`)

**What it measures:** Does the `GroupingAgent` correctly cluster a set of extracted
concepts into the same groups a human would, given the same meeting context?

The scorer produces two signals: `grouping.exact_match` (1.0 only if every predicted
group exactly matches every expected group) and `grouping.group_hit_rate` (fraction of
expected groups that appear exactly in the predicted set, giving partial credit).
Both are order-independent (frozenset comparison).

---

## Harness 5 — Verification (`VerificationEvalRunner`)

**What it measures:** Does the `VerificationAgent` correctly judge whether a node's
`quote` actually supports its `title`/`description` in the source transcript?

The scorer tallies a confusion matrix (TP/FP/FN/TN) per node; the runner aggregates
these into harness-level `precision` and `recall` metrics that feed the gate.

---

## Calibration meta-scorers

The meta-scorers in `eval/calibration/` are **threshold-tuning tools**, not eval
harnesses.  Their purpose is to find the optimal runtime threshold for a pipeline
signal (confidence cutoff, similarity cutoff) without re-running expensive LLM calls
on every iteration of development.

### File-cache-then-sweep pattern

Both meta-scorers share the same two-phase design:

1. **`_build_cache()`** — loads pipeline results for every corpus example using the
   shared file-based cache (`eval/cache.py` → `read_or_run`).  On a cache hit the
   result is read from disk instantly; on a miss the pipeline runs (LLM calls or
   embedding lookups) and the result is written to disk for future runs.  Cache files
   are keyed on `corpus_id` and agent fingerprint, so a prompt change automatically
   invalidates cached results for that agent.  Stale entries (corpus examples that no
   longer exist) are swept after every build.
2. **`sweep_threshold()`** — replays all in-memory results across thresholds in [0, 1]
   at `step` intervals (default 0.01) with no further I/O, returning a `SweepResult`
   with one `SweepPoint` per threshold and a `suggested_threshold` derived from the
   metric appropriate to each scorer (see per-scorer sections below).

The file cache is shared with the eval harnesses: if the identification or retrieval
harness has already run against the current corpus and prompts, the meta-scorer reuses
those results at no extra cost.

### `IdentificationMetaScorer`

Calibrates the `confidence_breakdown.heuristics` cutoff that routes identified nodes to
`APPROVED` (auto-accepted) vs. `PENDING_REVIEW` (human review queue).

**Important:** nodes below threshold are **not discarded** — they are deferred to the
review queue.  The threshold therefore controls how much of the pipeline output bypasses
human review, not how much is retained at all.  This has two consequences:

- **F1 is the wrong metric for `sweep_threshold`.**  F1 treats nodes below threshold as
  false negatives (missed items), but a deferred node is not a miss — it is just
  reviewed by a human instead.  The relevant metric is **precision among auto-approved
  nodes** paired with **coverage** (fraction of nodes auto-approved).  Pick the threshold
  where precision meets your acceptable false-positive rate; coverage tells you how much
  human review load remains.
- **The threshold is only meaningful relative to the confidence signal.**  With
  verification disabled, `final` equals the heuristics score alone.  The calibrated
  threshold will change once verification is enabled and weights are fitted — treat the
  heuristics-only threshold as provisional.

#### API

```python
# Exploratory: returns the full P/C curve across thresholds [0, 1]
# Use this to inspect the tradeoff and choose p_target
await scorer.precision_coverage_curve() -> list[IdentificationSweepPoint]

# Production: argmax coverage subject to precision_approved >= p_target (default 0.95)
# Falls back to argmax precision if no threshold meets p_target
await scorer.sweep_threshold(p_target=0.95) -> IdentificationSweepResult
```

Each `IdentificationSweepPoint` carries aggregate `precision_approved` and `coverage`,
plus a `per_type: dict[ConceptType, TypePC]` breakdown so per-type thresholds
(`ExtractionConfig.per_type_thresholds`) can be calibrated individually.

#### Calibration workflow

Both paths are driven by `calibrate identification` from the CLI.

```bash
# Optional: inspect the P/C curve and choose p_target
uv run python -m seshat.cli.eval.calibrate identification --pc-curve

# Sweep threshold (heuristics signal; works with or without verification)
uv run python -m seshat.cli.eval.calibrate identification

# Ignore verification gate results when sweeping (treats all nodes as gate-passed)
uv run python -m seshat.cli.eval.calibrate identification --ignore-verification
```

`sweep_threshold(p_target)` is the only calibration step. It sweeps
`confidence_threshold` across the corpus and selects the value that meets the
precision target `p_target` with maximum coverage. Because heuristics is always
the sole continuous signal, the sweep operates directly on `breakdown.heuristics`
— no weight fitting is required.

### `RetrievalMetaScorer`

Sweeps the `score_threshold` passed to `AbstractVectorStore.search`.  `_build_cache()`
loads each corpus fixture via the shared file cache.  Results are always seeded at
`score_threshold=None` (returning every candidate with its raw score), so a single
cached result supports replaying any threshold cutoff.  `sweep_threshold()` then
applies threshold filters in memory.

The suggested threshold is the argmax of **macro-F2** — the macro-average of F2 scores
across all corpus examples, where:

- **Positive examples** (`expected_relevant_ids` non-empty): contribute `F2 = (5 × P × R)
  / (4P + R)`.  F2 weights recall twice as heavily as precision (beta=2), reflecting that
  missing a relevant KB node is worse than surfacing an irrelevant one.
- **Negative examples** (`expected_relevant_ids` empty): contribute **specificity** (1.0
  if nothing is returned above threshold, 0.0 otherwise).  This prevents threshold=0
  from dominating: at threshold=0 every candidate passes, specificity collapses to 0
  for all negative examples, and macro-F2 is penalised accordingly.

Ties in macro-F2 resolve to the lower threshold (`np.argmax` on the ascending grid
returns the first occurrence).

`recall_at_5` and `precision_at_5` are still recorded on each `RetrievalSweepPoint`
for inspection and plotting, but they no longer drive threshold selection.

---

## `eval/cache.py` — `read_or_run`

```python
async def read_or_run(cache_file: Path, model_cls: type[M], coro: Coroutine) -> M
```

Used by the eval runners (grouping, verification) and the meta-scorers to avoid
re-running LLM calls across development iterations.  If `cache_file` exists it
deserialises and returns the cached `BaseModel` without awaiting the coroutine
(closing it cleanly to suppress warnings).  Otherwise it awaits the coroutine and
writes the result as JSON.

`clear_cache_dir(cache_dir)` deletes all `.json` files in a directory; runners call
this at the end of a successful run so the next invocation always produces fresh
predictions.
