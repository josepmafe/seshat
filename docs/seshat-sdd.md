# Seshat — Solution Design Document

Translates Seshat's high-level architecture into an implementation-oriented blueprint. Intended for engineers who will build, test, and operate the system, and for reviewers validating coherence and feasibility.
- For a narrative introduction see [docs/primer.md](primer.md).
- For a quick-reference decision index see [docs/architecture.md](architecture.md).
- For full implementation contracts see [docs/superpowers/specs/2026-04-21-seshat-design.md](superpowers/specs/2026-04-21-seshat-design.md).

---

## Purpose & Scope

**In scope**

- Concrete description of components, responsibilities, and boundaries.
- Core data models and contracts that glue components together.
- End-to-end control flows for key scenarios.
- LLM/agent orchestration approach.
- Algorithms and heuristics that materially affect system behaviour.
- Security, safety, observability, and evaluation hooks.

**Out of scope**

- Exhaustive API parameter lists, enum values, and corner cases (covered by the design spec).
- Detailed DB schema migrations and ORM mappings.
- Deployment infrastructure (Docker, CI/CD, cloud-specific configuration).

---

## System Overview

Seshat is an API-first GenAI application that ingests technical meeting recordings or pre-formatted transcripts, extracts structured decisions, risks, open questions, and action items via a multi-agent LLM pipeline, and persists the results to a graph-shaped knowledge base backed by Postgres and pgvector. A Streamlit UI sits on top of the API to let reviewers submit meetings, monitor progress, and approve or reject extracted nodes.

```
Streamlit UI → FastAPI → Pipeline Worker → Storage Layer
                  ↑                              │
                  └──────────── Results ─────────┘
```

**Streamlit UI** — thin client for job submission, progress tracking, and review. Communicates exclusively with the FastAPI service.

**FastAPI Service** — authenticates requests, validates inputs, owns the external API contract, enqueues work onto the task queue, and exposes job status for polling.

**Pipeline Worker** — consumes jobs from the task queue and executes a sequential pipeline: ingestion, chunking, extraction, RAG + resolution, confidence scoring, review gating, and storage. Encapsulates all orchestration logic around multi-agent LLM calls.

**LLM & Agent Layer (LangChain)** — implements extraction and verification agents as LangChain chains/tools. Interfaces with multiple LLM providers via a configuration-driven abstraction. Leverages prompt caching and MLflow autologging.

**Storage Layer** — Postgres (`ops` schema) for jobs, API keys, KB nodes, and relationships; pgvector (separate schema) for dense embeddings; S3-compatible blob store (LocalStack for dev) for raw audio, transcripts, and extraction artifacts.

**Observability** — MLflow v3.x as the observability backbone, integrated with LangChain via `mlflow.langchain.autolog()` for per-agent traces and metrics.

---

## Component Responsibilities & Boundaries

### Streamlit UI

**Responsibilities**

- Guided flow for uploading audio or pasting pre-formatted text, configuring job options, displaying job status (`PENDING`, `TRANSCRIBING`, `WRITING`, `AWAITING_REVIEW`, `DONE`, `FAILED`), and rendering extracted nodes for human review.
- Captures reviewer decisions (approve / reject / edit) and sends them to the API.

**Interactions**

- `POST /jobs` — submit work.
- `GET /jobs/{id}` — poll status and retrieve nodes pending review.
- `POST /jobs/{id}/approve` — submit review decisions.
- Does not talk directly to the worker, storage layer, or LLM providers.

**Error handling / UX**

- Polling-based progress updates; no push notifications in MVP.
- Clear empty states for: no nodes available yet; job failed (surface reason summary from API).
- No long-lived WebSocket/SSE connections in MVP.

---

### FastAPI Service

**Authentication & authorisation**

- Validate API keys from `X-API-Key` header.
- Roles: `viewer` (read-only), `reviewer` (review jobs), `operator` (auto-mode and operations endpoints), `admin` (delete operations + all operator actions).
- Root API key (stored in Secrets Manager under `APIConfig.admin_api_key_secret_key`) gates the `/admin` router for key management.

**Request validation**

- Audio path: streaming size check against `TranscriptionConfig.max_file_bytes`; magic-byte validation for allowed formats (MP3, WAV, M4A).
- Text path: structural validity check for YAML/JSON inputs.
- Reject invalid inputs early with appropriate HTTP status codes.

**Job lifecycle API**

- `GET /jobs` — list jobs with optional `job_status` filter, pagination (`limit`, `offset`).
- `POST /jobs` — enforce per-user rate limiting and global concurrency cap; enforce idempotency via `idempotency_key`; persist initial job record in Postgres; enqueue work via `AsyncioTaskQueue`. Accepts `auto_mode` top-level flag as a shorthand for operator-role auto-approval.
- `GET /jobs/{id}` — return job status and timestamps (`created_at`, `updated_at`, `finished_at`).
- `POST /jobs/{id}/approve` — accept `bulk_rules` (applied first) and explicit per-node decisions (applied second); transition job state (`AWAITING_REVIEW → WRITING → DONE`).

**Admin API** (`/admin`, root-key authenticated)

- `GET /admin/api-keys` — list all API keys with revocation status.
- `POST /admin/api-keys` — create a new API key; plaintext returned once.
- `DELETE /admin/api-keys/{key_id}` — revoke an API key.

**Task queue interaction**

Uses a task queue abstraction: `enqueue(fn, *args, **kwargs) → job_id`, `get_status(job_id) → JobStatus`, `cancel(job_id) → bool`. Queue implementation (asyncio vs ARQ/Redis) is not visible to API handlers beyond the interface.

**Boundaries**

- Does not call LLMs or implement pipeline stages.
- Does not access KB node/relationship tables directly except for job metadata where necessary.
- Treats the worker as the sole owner of pipeline execution logic.

---

### Pipeline Worker

Dequeues jobs from `AsyncioTaskQueue` and orchestrates the pipeline stages. Maintains job status transitions and side effects on Postgres and blob storage. Enforces the two-pass extraction + RAG pattern. Integrates with MLflow for metrics and traces.

**Pipeline stages**

1. **Ingestion & Transcription** — for audio, reads from blob store and calls the transcription provider to produce a `TranscriptDocument`. For pre-formatted text, validates and normalises into `TranscriptDocument`.

2. **Chunking** — applies TextTiling (NLTK) to segment the transcript into topical chunks. If the chunking sanity check (§12) indicates systematic mis-segmentation, falls back to fixed-size overlapping windows (500-token windows, 100-token overlap).

3. **Pass 1 — Multi-Agent Extraction** — for each chunk and each `ConceptType`, runs the corresponding extraction agent concurrently. Collects `KBNode` candidates with `relationships: []`. Action item agents additionally record `assignee: str` (required; ownerless items are not emitted — see Known Limitations).

4. **Within-Meeting Deduplication** — merges nodes of the same type within a single meeting: exact title match (primary), then embedding similarity ≥ 0.85 (fallback). No `SUPERSEDES` relationship is created within a single job.

5. **Pass 2 — RAG + Resolution** — uses the deduplicated node set as the working candidate list. Queries the existing KB and vector store to attach `KBRelationship` objects (`SUPERSEDES`, `AMENDS`, `CONFLICTS_WITH`, `DEPENDS_ON`) and resolve action item assignees against `TranscriptMetadata.participants`. RAG runs after extraction; extraction agents receive only a lightweight hint context.

6. **Confidence Scoring** — heuristics signal (spaCy) is the sole continuous confidence signal. Verification is a hard binary gate: a failed verification rejects the node regardless of its heuristics score.

7. **Review Gating / WRITING** — decides per-node whether to auto-approve (operator role + auto-mode, or high confidence + policy) or route to human review (`AWAITING_REVIEW`). Writes `curated/extraction.json` at the start of `WRITING` so the artifact exists even if all nodes are later rejected.

8. **Storage & Finalisation** — writes approved nodes and their vector embeddings in a single Postgres transaction. Marks job as `DONE` (possibly with an empty result set) or `FAILED` (with `recoverable` flag where appropriate).

**Boot-time recovery**

On startup, detects jobs stranded in `WRITING` and marks them `FAILED(recoverable=True)` before accepting new work.

**Boundaries**

- Owns all job lifecycle state transitions.
- Owns all calls to LLM providers and vector/KB stores.
- Does not expose a public network interface in MVP; interacts with the API only via the queue and shared DB.

---

### Storage Layer

**`PostgresKBStore`** — `kb_nodes` and `kb_relationships` tables in the `ops` schema. Concrete class — no abstract base; single MVP implementation. Provides insert/update and query-by-id, meeting, type, or relationship. KB rows and their associated embeddings are written in a single transaction.

**`OpsLedger`** — owns all reads and writes to `ops.jobs` and `ops.api_keys`. Exposes typed CRUD methods; no raw SQL leaks into higher layers. `finished_at` is set on terminal transitions (`DONE`, `FAILED`) and cleared on `reset_failed_job`.

**`PGVectorStore`** — pgvector in a separate `store` schema, accessed via `langchain-postgres`. Stores embeddings and metadata (node id, concept type). Used for both RAG retrieval in Pass 2 and deduplication similarity checks.

**`S3BlobStore`** — S3-compatible storage (LocalStack in MVP, AWS S3 in production). Concrete class — no abstract base; single MVP implementation. Path layout is date + job-ID based for human-readable, chronologically browsable paths — see the design spec §2 for the full structure.

Exact paths are centralised in config to avoid scatter.

**Boundaries**

KB store and vector store share the same Postgres instance but remain logically independent (different schemas). Blob store is accessed directly via `S3BlobStore` from pipeline stages.

---

### Cross-Cutting Concerns

**Configuration** — single process-wide `SeshatConfig` built on pydantic-settings with `env_nested_delimiter="__"`. Only the root config model inherits from `BaseSettings` to prevent dual env-var resolution in nested models. Per-request overrides are deep-merged onto the singleton into a new object; the singleton is never mutated. API-specific settings (`max_jobs_per_user_per_hour`, `max_concurrent_jobs`, eval gate path, startup skip flags, admin key name) live under `SeshatConfig.api` (`APIConfig`). Logging settings (root level, per-logger overrides) live under `SeshatConfig.logging` (`LoggingConfig`).

**Secrets** — `AbstractSecretsProvider` with `EnvSecretsProvider` (local/MVP) and `AWSSecretsProvider` (cloud). Secrets are resolved once at startup and cached in-process. Rotations require a worker restart in MVP; v2 adds TTL-based refresh.

**Task queue** — MVP: in-process `AsyncioTaskQueue`. Contract: `enqueue(coro, *args) → job_id`, `get_status() → JobStatus`, `cancel() → bool`. v2: `ARQTaskQueue` with durable Redis-backed semantics at the same call-sites.

---

## Key Data Models & Contracts

This section captures only the "spine" models used to connect components. Full fields and validations are in the design spec.

**Job** — `job_id: UUID`, `status: JobStatus` (`PENDING`, `TRANSCRIBING`, `EXTRACTING`, `AWAITING_REVIEW`, `WRITING`, `DONE`, `FAILED`), `source_type` (audio/text), `created_at`, `updated_at`, `finished_at` (set on terminal states), `idempotency_key: str | None`, `user_id`, `meeting_date`, `raw_blob_key`, `submission` (JSONB). Persisted in `ops.jobs`.

**`TranscriptDocument`** — `job_id: UUID`, `raw_text: str`, token count metadata, segments/chunks (when attached), `metadata: TranscriptMetadata` (participants, meeting title, date, optional tags).

**`KBNode`** — `id: UUID`, `job_id: UUID`, `concept_type: ConceptType` (`DECISION`, `RISK`, `ACTION_ITEM`, `OPEN_QUESTION`), `title: str`, `content: str`, `source_quote: str`, `confidence: float`, `assignee: str` (action items only; required — see Known Limitations), `due: str | None` (action items only).

**`KBRelationship`** — `id: UUID`, `from_node_id: UUID`, `to_node_id: UUID`, `relationship_type` (`SUPERSEDES`, `AMENDS`, `CONFLICTS_WITH`, `DEPENDS_ON`, `ASSIGNED_TO`).

**Job submission request** — `input_type`, file (binary) or body (YAML/JSON), optional config overrides, `idempotency_key`.

**Job status response** — `job_id`, `status`, `created_at`, `updated_at`, `finished_at` (set on terminal states), `idempotency_key`, `stage_progress`, `error`, `mlflow_run_id`.

**Approval request** — `bulk_rules` (threshold rules, processed first) and `decisions` (list of `{node_id, action: approve|reject, edited_content?, edited_title?}`, processed second).

---

## Control Flows

### Job Submission & Processing (Happy Path)

1. User selects an audio file in Streamlit and clicks "Submit".
2. Streamlit sends `POST /jobs` with the file stream and configuration.
3. FastAPI authenticates, applies rate limiting and concurrency checks, validates file size (streaming — aborts with HTTP 413 on threshold breach) and magic bytes, persists a new job row in Postgres (`PENDING`), stores the input file via `S3BlobStore`, enqueues a pipeline task, and returns job id and initial status.
4. Pipeline Worker picks up the job and advances through stages: `TRANSCRIBING` → `EXTRACTING` → Pass 1 extraction → within-meeting dedup → Pass 2 RAG + resolution → confidence scoring → writes `curated/extraction.json` → either proceeds directly to `WRITING` + `DONE` (auto-approve policy) or transitions to `AWAITING_REVIEW`.
5. Streamlit polls `GET /jobs/{id}` until status becomes `AWAITING_REVIEW` or `DONE`.

### Review & Approval Flow

1. `GET /jobs/{id}` returns `AWAITING_REVIEW` with nodes pending review, confidence scores, and minimal context.
2. Reviewer inspects and edits nodes in Streamlit, selects approve/reject decisions, and submits `POST /jobs/{id}/approve` with optional `bulk_rules` and explicit per-node decisions.
3. FastAPI validates the payload and transitions job state to `WRITING`.
4. Worker writes all approved nodes and relationships to `PostgresKBStore` (KB tables) and `PGVectorStore` (embeddings) in a transactionally consistent way.
5. Job transitions to `DONE` — with non-empty results if any nodes were approved, with an empty result set if all were rejected (still a successful run).

### Failure & Recovery Cases

**Transcription error** — worker marks job `FAILED` with a human-readable error reason; no KB or vector writes occur. UI surfaces the failure with a reason summary.

**Worker crash during `WRITING`** — on next startup, query jobs in `WRITING` state and mark them `FAILED(recoverable=True)`. Avoids jobs stuck in `WRITING` indefinitely.

**Idempotent resubmission** — if `POST /jobs` is retried with the same `idempotency_key`: existing job still in-progress or completed → return existing job id and status; existing job `FAILED` → start a fresh job and record the linkage for traceability.

---

## LLM & Agent Orchestration

**Providers & routing**

- Primary extraction provider and verification provider are set via `SeshatConfig`.
- Verification provider must differ from the extraction provider (enforced by `model_validator`) to avoid correlated failures.
- Weakest valid configuration: extraction provider set (e.g. Anthropic), verification agent disabled → heuristics-only scoring with a startup warning.

**Agent registry**

Central registry maps `ConceptType` → agent implementation:

| `ConceptType` | Agent |
|---------------|-------|
| `DECISION` | `DecisionIdentificationAgent` |
| `RISK` | `RiskIdentificationAgent` |
| `ACTION_ITEM` | `ActionItemIdentificationAgent` |
| `OPEN_QUESTION` | `OpenQuestionIdentificationAgent` |

Adding a concept type: implement an agent class (inheriting `_BaseIdentificationAgent`) and add it to the registry.

**Pass 1 — Extraction**

For each chunk and each registered concept type, constructs a prompt using a standardised layout: `## Definition` (concept boundary), `## Task` (extraction instruction with hard stops), `### Field identification rules` (one bullet per output field), `## Over-extraction guards` (logical binary tests + typed counter-examples), `## Boundary examples` (positive and negative per pair), and `## Positive criteria`. Agents produce `KBNode` candidates with `relationships: []` and an optional `assignee` for action items. Non-conforming responses are rejected and optionally retried.

When `resolution_self_review.enabled` is `True`, each identification agent is wrapped in `ReflectiveIdentificationAgent`, which adds an **extract → validate → filter** pass: after extraction, a single validation call checks each item for logical compliance (does it satisfy the extraction rules?) and semantic compliance (does the description match the quote?). Items that fail are discarded. The validator is conservative by design — it rejects only clear rule violations, not borderline quality. On any validation failure (retries exhausted, count mismatch), all extracted nodes are returned as-is.

**Pass 2 — RAG + Resolution**

After collecting all Pass 1 nodes: embeds new nodes, queries `PGVectorStore`, and uses retrieval results plus transcript context to attach `KBRelationship` entries for cross-meeting links and resolve `ASSIGNED_TO` by matching `assignee` strings against participants. Nodes in transient states (`WRITING`) are excluded from retrieval context.

When `resolution_self_review.enabled` is `True`, same-type resolution agents are wrapped in `ReflectiveResolutionAgent`, which adds a **competing-hypothesis tiebreaker** for ambiguous cases. The inner agent signals uncertainty via an optional `alt_rel_type` field — populated only when two relationship types are genuinely competing for a pair. Only contested entries are sent to a tiebreaker call; unambiguous entries bypass it entirely. This design recovers the recall penalty of a blanket validate-and-filter approach while keeping token overhead near the shallow baseline. Cross-type resolution agents are not wrapped — eval results show no quality gap there.

**Prompt caching**

- OpenAI: automatic prefix caching for long system prompts.
- Anthropic: explicit `cache_control` headers on the system prompt block.
- The LLM wrapper encapsulates the caching strategy; agents assemble prompts but do not manage caching themselves.

---

## Algorithms & Heuristics

### Chunking

Default: TextTiling (NLTK), tuned for long-form transcripts. If the chunking sanity check (§12) indicates systematic over- or under-segmentation, falls back to fixed-size overlapping windows (500-token windows, 100-token overlap).

### Within-Meeting Deduplication

1. Group nodes by `concept_type`.
2. Within each group: merge nodes with identical titles (case-insensitive); for non-exact matches, merge if embedding similarity ≥ `merge_similarity_threshold` (0.85).
3. When merging: aggregate source quotes if configured; drop earlier nodes. No `SUPERSEDES` relationship is created within a single job.

### Confidence Scoring

Heuristics (spaCy) is the sole continuous signal; `KBNode.confidence` equals the heuristics score. Verification is a hard binary gate: when a `VerificationLLMConfig` is configured, nodes that fail verification are rejected outright regardless of heuristics score. Full heuristics formula is defined in [docs/superpowers/specs/2026-04-21-seshat-design.md](superpowers/specs/2026-04-21-seshat-design.md).

### Threshold Calibration

`src/seshat/eval/calibration/` provides two meta-scorers for empirically tuning the parameters above:

- **`IdentificationMetaScorer`** — sweeps `confidence_threshold` (the auto-approval cut-off) across the eval corpus via `sweep_threshold(p_target)`, reporting precision/coverage curves so the optimal threshold can be read off and committed to config.
- **`RetrievalMetaScorer`** — sweeps the vector similarity threshold used by `NodeRetriever` to tune the precision/recall tradeoff for the retrieval step.

Both emit `SweepResult` objects and log to MLflow. Recalibrate any time agent prompts or model provider/version change.

---

## Security & Safety

**Authentication & authorisation** — API key in `X-API-Key` header; keys stored hashed with bcrypt (cost factor 12). Roles: `viewer` (read-only access to jobs and graph), `reviewer` (submit and review jobs), `operator` (auto-mode, retry, graph mutations), `admin` (delete operations + all operator actions). A separate root key gates the `/admin` router for key provisioning and revocation.

**Rate limiting & concurrency** — per-user hourly job cap and global concurrency cap, both enforced at `POST /jobs`. Capped jobs may be rejected with 429 or queued depending on future configuration.

**Prompt injection mitigation** — see [docs/superpowers/specs/2026-04-27-prompt-interaction-design.md §3](superpowers/specs/2026-04-27-prompt-interaction-design.md) for the full security model (structural isolation, output validation, source quote verification, context sanitisation) and per-agent coverage table.

**Data handling & secrets** — secrets resolved via `AbstractSecretsProvider` at startup and cached in-process. Transcripts, prompts, and responses written to MLflow are considered sensitive; access to the MLflow tracking server must be controlled accordingly. LocalStack emulates blob storage and Secrets Manager in local development.

---

## Observability & Evaluation

### MLflow Integration

`mlflow.langchain.autolog()` instruments all LangChain agent calls. Captured per-agent data: agent identity (concept type, pass), LLM usage (tokens, cost, audio seconds), prompt cache hit/miss, latency, errors and retries, and input/output artifacts (subject to sensitivity constraints).

Run organisation: each pipeline job groups related agent calls under a single MLflow run, tagged with environment (`dev`, `test`, `prod`) and `job_id` for correlation with Postgres.

### Operational Metrics

At minimum, the following metrics should be emitted:

- Per-stage latency (transcription, chunking, extraction, resolution, writing).
- Job throughput (jobs/hour) and outcome rates (success, failure, empty result).
- Queue depth and average wait time.
- LLM error rates by provider.
- Confidence score distributions per concept type.

### Release Gate & Evaluation Harness

The eval harness runs five independent passes, each with its own corpus under `data/eval/corpus/<pass>/`, runner, scorer, and gate targets. Passes are togglable via `EvalConfig` and can be run individually; `upsert_gate` carries over blocks from the existing file so a partial run only updates what it ran.

**Identification pass** — extraction quality. Per-concept-type precision, recall, spurious rate against quote-anchored ground truth. Additional field-level accuracy scores (assignee, due, rationale, risk type) are logged to MLflow as observability signals but are not gated.

**Resolution pass** — relationship inference quality. Corpus cases supply source nodes, KB nodes, and expected `(source, target, rel_type)` triples; scorer does exact triple match → per-concept-type precision and recall.

**Retrieval pass** — vector search quality. Corpus cases supply a query node, candidate pool, and expected relevant IDs; scorer measures recall@5 (gated) and precision@5 (logged).

**Verification pass** — verification agent quality. Precision and recall against ground-truth accept/reject decisions.

**Grouping pass** — grouping agent quality. Group hit rate (gated) and exact match (logged).

**Gate file** (`data/eval_gate.json`) — `GateResult` with five metric blocks (`identification_metrics`, `resolution_metrics`, `retrieval_metrics`, `verification_metrics`, `grouping_metrics`) plus a computed `passed` field. A `None` block means the pass was not run and is not a failure; `passed` is `false` if all blocks are `None`. The worker refuses to accept jobs at startup unless the gate file is present and `passed=true`.

**Regression gate:** any change to agent system prompts, model provider/version, or confidence scoring heuristics must be accompanied by a passing eval run (at minimum the affected passes) that updates `data/eval_gate.json`. Gate thresholds are centralised in `src/seshat/eval/thresholds.py`.

## Known Limitations

### Action Item: ownerless tasks are not captured

The `ActionItem` model enforces `assignee: str` (non-nullable). The identification agent prompt requires an identifiable owner before emitting — anonymous self-references and explicitly unowned work are suppressed. As a result, legitimate follow-up tasks that emerged from a meeting without a named owner (e.g. "someone from the platform team needs to handle this") are silently dropped rather than captured with a null assignee.

**Workaround:** none currently. Downstream consumers should be aware that the action item list may be incomplete for meetings where ownership was discussed but not formally assigned. A future improvement could introduce a separate concept type or a low-confidence ownerless task variant, but this is deferred until there is evidence of user need from real transcripts.
