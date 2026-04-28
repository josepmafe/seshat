# Seshat Spec — Review TODO List

Generated from genai-review panel (9 agents + devil's advocate + tech lead synthesis).
Each item is a spec update required before the design is implementation-ready.

---

## Tier 1 — Must Fix Before MVP Ships

These are blocking. The system cannot be safely deployed without addressing them.

---

### ~~AUTH-1: Add authentication and authorization to all API endpoints~~ ✓ REVIEWED
**Source:** Security Expert (Critical), Domain SME (Critical)
**Evidence:** Section 8 — no auth described on any endpoint
**Resolution:** API key in `X-API-Key` header. Keys stored as bcrypt hashes in SQLite (`api_keys` table). Three roles: `submitter`, `reviewer`, `operator`. Provisioned via `seshat create-api-key` CLI. JWT deferred to v2. Added to Section 8.

---

### ~~AUTH-2: Add access controls to S3 bucket~~ ✓ REVIEWED — N/A
**Source:** Security Expert (Critical), Solutions Architect (Important), Data Engineer (Critical)
**Evidence:** Section 2 S3 Artifact Storage — no bucket policy, IAM, encryption, or lifecycle rules
**Resolution:** Out of scope — system runs locally against LocalStack only (master's thesis). A scope note added to Section 2. Production hardening deferred to if/when this is deployed to real AWS.

---

### ~~SECURITY-1: Define prompt injection mitigation strategy~~ ✓ REVIEWED
**Source:** Security Expert (Critical)
**Evidence:** Section 4 — transcript text and retrieved KB context injected into agent system prompts without sanitization
**Resolution:** Added "Prompt Injection Mitigation" subsection to Section 4: structural isolation via delimited sections, output validation against `KBNode` schema, KB context sanitisation through Pydantic, second-order risk note.

---

### ~~SECURITY-2: Validate LLM responses before downstream use~~ ✓ REVIEWED
**Source:** Security Expert (Important)
**Evidence:** Section 4 — agent outputs flow into KB with no schema enforcement described
**Resolution:** Covered by SECURITY-1 point 2 — output validation against `KBNode` schema is part of the prompt injection mitigation section added to Section 4.

---

### ~~CHUNK-1: Fix chunking strategy for MVP — speaker turns are None~~ ✓ REVIEWED
**Source:** SW Engineer (Important), Data Engineer (Important), AI System Designer (Critical)
**Evidence:** Section 4 Chunking says "split by speaker turns"; Section 2 TranscriptMetadata — `turns: list[Turn] | None` is reserved for v2 (always None in MVP)
**Resolution:** Replaced with TextTiling (NLTK). Topic-shift boundary detection, no model calls, variable-length coherent chunks. `max_chunk_count` in `ExtractionConfig` as hard ceiling. v2 paths: diarization-based splitting or semantic chunking, to be evaluated against eval corpus.

---

### ~~CHUNK-2: Chunking semantic unit should be concept-aligned, not transcript-aligned~~ ✓ REVIEWED
**Source:** AI System Designer (Important — standalone from CHUNK-1)
**Evidence:** Section 5 ChunkMetadata — `node_type` and `node_id` imply 1:1 chunk-to-concept mapping; Section 4 Chunking uses transcript turns as the splitting unit
**Resolution:** Renamed `ChunkMetadata` → `NodeMetadata` throughout. Added clarifying note to Section 5: `NodeMetadata` describes an extracted node, not a raw transcript segment; vector store indexes one vector per `KBNode`. Removed `chunk_index` field (transcript artifact). Two-stage model (transcript windows for agents vs. node embeddings for RAG) made explicit.

---

### ~~COST-1: Define per-job token budget and cost ceiling~~ ✓ REVIEWED
**Source:** FinOps (Critical), AI/ML Engineer (Important)
**Evidence:** Section 4 — 4 agents run in parallel against full transcript with no token ceiling on extraction prompts; Section 5 `max_context_tokens=4000` applies to RAG context only
**Resolution:** Added `max_chunk_count=50`, `max_output_tokens_per_agent_call=2048`, `max_budget_usd=2.50` (USD) to `ExtractionConfig`. Rough estimate: ~$1.50 per 15K token transcript at gpt-4o pricing; $2.50 gives ~1.7× headroom for Anthropic models.

---

### ~~COST-2: Add cost alerting and budget cap mechanism~~ ✓ REVIEWED
**Source:** FinOps (Critical)
**Evidence:** Section 9 — MLflow tracks cost per agent call but no threshold or alert mechanism exists
**Resolution:** Covered by COST-1 — `max_budget_usd` added to `ExtractionConfig`. Worker transitions job to `FAILED` if exceeded.

---

### ~~COST-3: Add prompt caching for static agent system prompts~~ ✓ REVIEWED
**Source:** FinOps (Important)
**Evidence:** Section 4 Agent Registry — each agent has a static system prompt reused across every job call; no caching strategy mentioned
**Resolution:** Added "Prompt Caching" subsection to Section 4. OpenAI automatic prefix caching, Anthropic `cache_control` headers. Flagged as hard constraint on orchestration framework choice.

---

### ~~AUTO-1: Add authorization and audit trail to auto_mode~~ ✓ REVIEWED
**Source:** Data Engineer (Important), Domain SME (Critical)
**Evidence:** Section 3 ExtractionConfig — `auto_mode: bool = False`; Section 3 Per-Request Overrides — `SeshatConfigOverride` includes `extraction`, so any caller can enable `auto_mode`
**Resolution:** `auto_mode=True` restricted to `operator` role. MLflow audit log includes `user_id`, timestamp, job ID, and auto-approved node list. Added to Section 8.

---

### ~~FAILED-1: Define FAILED state recovery path and error payload~~ ✓ REVIEWED
**Source:** SW Engineer (Critical), UX Designer (Critical)
**Evidence:** Section 8 Job Status Model — `FAILED` is a terminal state with no retry, partial result, or error payload defined
**Resolution:** Added `ErrorPayload` model (`job_id`, `stage`, `reason`, `recoverable`, `tokens_used`). Token tracking per stage; cost computed at display time from price table. Per-step `max_budget_usd` caps in `ExtractionConfig` and `TranscriptionConfig`. `POST /jobs/{id}/retry` with operator-only cap overrides. Recoverable vs fatal failure modes defined. UI retry button surfaces when `recoverable=True`.
**Post-review update (2026-04-24):** `ErrorPayload.job_id` removed — it was redundant with `JobResponse.job_id` (the only carrier of `ErrorPayload`). `usage` field retyped from `dict[str, ...]` to `dict[JobStatus, list[UsageRecord]]` for type safety.

---

## Tier 2 — Must Define Before Implementation Starts

These are not blocking for safety, but implementation cannot proceed coherently without them.

---

### ~~EVAL-1: Define minimum viable evaluation strategy~~ ✓ REVIEWED
**Source:** AI/ML Engineer (Critical), Domain SME (Critical)
**Evidence:** Section 9, Decisions Deferred — no eval dataset, no ground-truth, no correctness metric; confidence threshold (0.7) has no calibration basis
**Resolution:** Added Section 12 "Evaluation Strategy". Hand-crafted synthetic corpus in `tests/eval/corpus/`. Precision/recall targets per `ConceptType`. Threshold calibration via PR-curve sweep. `seshat eval` CLI command wraps `mlflow.genai.evaluate()` — eval runs are versioned MLflow experiments. Regression gate: run `seshat eval` before promoting any prompt/model/scoring change.

---

### ~~EVAL-2: Address verification agent confirmation bias risk~~ ✓ REVIEWED
**Source:** Domain SME (Critical — corrected by DA from "closed loop" to "shared model bias")
**Evidence:** Section 4 Confidence Scoring — verification agent and extraction agent may share model family, leading to correlated errors
**Resolution:** Added `VerificationConfig(provider: LLMProvider, model: str)` nested inside `ExtractionConfig`. Cross-provider constraint documented in Section 4 Confidence Scoring: verification provider must differ from `LLMConfig.extraction_provider`. Enforced by convention at config time.

---

### ~~MODEL-1: Specify model choices~~ ✓ REVIEWED
**Source:** AI/ML Engineer (Important), FinOps (Important), AI System Designer (Important)
**Evidence:** Section 4, Section 5 — no model named anywhere; extraction agent tier, embedding model, and verification model are all unspecified
**Resolution:** Defined `LLMConfig(extraction_provider, extraction_model="claude-sonnet-4-6", concept_model_overrides)` in Section 3. `RAGConfig` expanded with `embedding_model="text-embedding-3-small"` and `rerank_model`. Per-`ConceptType` model overrides via `concept_model_overrides: dict[ConceptType, str] | None`. Verification model lives in `VerificationConfig` (EVAL-2).
**Post-review update (2026-04-24):** Two YAGNI knobs dropped. `concept_model_overrides` removed — no eval data yet motivates per-concept-type model tuning; add back when eval corpus results show it's needed. `rerank_model` and `rerank_provider` removed, `RerankProvider` enum deleted, reranking deferred to v2 (see new row in Decisions Deferred). `top_n` collapsed into `top_k=5`. Rationale: the retrieval baseline (RAG-1) measures recall@5 from vector search alone; rerank is an add-if-needed, not a default.

---

### ~~QUEUE-1: Resolve queue system choice~~ ✓ REVIEWED
**Source:** SW Engineer (Critical), Solutions Architect (Critical)
**Evidence:** Section 1, TBD table — queue system is undefined but every job state transition depends on it
**Resolution:** Python `asyncio` task queue for MVP (zero new infra). Durability limitation documented. `AbstractTaskQueue` factory interface required — ARQ/Redis-compatible contract so v2 swap is a provider change, not a refactor. TBD table entry resolved. Section 1 Architecture updated.

---

### ~~IDEMPOTENCY-1: Add job idempotency key and resume-from-checkpoint~~ ✓ REVIEWED
**Source:** Data Engineer (Critical)
**Evidence:** Section 2 S3 path, Section 8 POST /jobs — no idempotency key; re-submitting the same file creates a new job
**Resolution:** `idempotency_key: str | None` added to `TranscriptDocument` and `POST /jobs`. Deduplication logic defined: existing non-FAILED job → return existing ID; FAILED job + `resume=True` → reuse raw S3 artifacts. Documented in new "Job Idempotency" subsection in Section 8.

---

### ~~CONSISTENCY-1: Define KB write + vector upsert consistency strategy~~ ✓ REVIEWED
**Source:** Data Engineer (Critical), SW Engineer (Important)
**Evidence:** Section 6 — two independent factories; Section 2 S3 — two independent writes. No transaction boundary or reconciliation
**Resolution:** Write order defined in new "Write Order and Consistency" subsection in Section 6: KB first (source of truth), vector upsert second, failed upserts logged to retry queue. Same pattern for S3. `GET /jobs/{id}/sync-status` added as diagnostic endpoint (Section 8 endpoints table).

---

### ~~SCHEMA-1: Add schema version to KBNode, NodeMetadata, TranscriptDocument~~ ✓ REVIEWED
**Source:** Data Engineer (Critical)
**Evidence:** Section 6 MVP Markdown + Git — YAML frontmatter will be deserialized back; no version field protects against silent breakage on model evolution
**Resolution:** `schema_version: str = "1.0"` added to `KBNode`, `NodeMetadata`, and `TranscriptDocument`. Migration note added to Section 6 MVP Markdown + Git: `MarkdownKBStore` rejects nodes with unknown/incompatible schema version on read — hard failure preferred over silent deserialization.

---

### ~~ASYNC-1: Resolve sync vs async contract on storage ABCs~~ ✓ REVIEWED
**Source:** SW Engineer (Critical)
**Evidence:** Section 6 Interfaces — `AbstractKBStore` and `AbstractVectorStore` methods declared as plain `def`, not `async def`; pipeline runs in an asyncio context
**Resolution:** All `AbstractKBStore` and `AbstractVectorStore` methods changed to `async def` in Section 6. Note added: blocking implementations must use `asyncio.to_thread()`; async-native clients implement directly.

---

### ~~RAG-1: Define retrieval quality measurement baseline~~ ✓ REVIEWED
**Source:** AI/ML Engineer (Important), AI System Designer (Critical)
**Evidence:** Section 5 RAGConfig — `top_k=20`, `top_n=5` with no rationale or metric; no recall@k or MRR defined anywhere
**Resolution:** "Retrieval Quality Baseline" subsection added to Section 5. Test KB seeded from eval corpus via `seshat init`. recall@5 metric defined against known-relevant nodes. `seshat eval` runs retrieval baseline alongside extraction eval in the same MLflow run.

---

### ~~RAG-2: Define graph traversal depth and scope~~ ✓ REVIEWED
**Source:** AI System Designer (Important)
**Evidence:** Section 5 RAG step 4 — "Graph traversal on top-N nodes" with no depth, relationship types, or expansion cap
**Resolution:** `traversal_max_depth: int = 1` and `traversal_rel_types: list[RelationshipType] | None = None` added to `RAGConfig`. Risk of unbounded traversal inflating context documented in Section 5 note.

---

### ~~CONTEXT-1: Define end-to-end context window budget~~ ✓ REVIEWED
**Source:** AI/ML Engineer (Important), AI System Designer (Important), FinOps (Important)
**Evidence:** Section 5 `max_context_tokens=4000` caps RAG context only; total prompt = system prompt + RAG context + transcript chunk + output schema — no combined ceiling
**Resolution:** `max_transcript_chunk_tokens: int = 8000` added to `ExtractionConfig`. Full prompt budget documented: `system_prompt (~500t) + retrieved_context (≤4000t) + transcript_chunk (≤8000t) + output_schema (~200t) ≤ 12700t` — well within claude-sonnet-4-6's 200k window. Practical ceiling is cost, not context limit.

---

## Tier 3 — Design Refinements

Necessary but not blocking for safety or implementation start.

---

### ~~MODEL-2: Replace KBNodeEdit with narrower type for NodeDecision.edited_content~~ ✓ REVIEWED
**Source:** SW Engineer (Important), Data Engineer (Important)
**Evidence:** Section 8 Review Flow — `edited_content: KBNode | None` allows overwriting `id`, `confidence`, `status`, `relationships`, `metadata`
**Resolution:** Added `KBNodeEdit(title, description)` to Section 8. `source_quote` excluded — it is provenance, not interpretation; editing it would silently misrepresent evidence. `NodeDecision.edited_content` changed to `KBNodeEdit | None`. System-generated fields are immutable post-extraction.

---

### ~~MODEL-3: Add source_id to KBRelationship~~ ✓ REVIEWED
**Source:** SW Engineer (Important)
**Evidence:** Section 6 Interfaces — `write_relationship(self, rel: KBRelationship)` has no source; Section 4 `KBRelationship` — only `target_id` and `rel_type`
**Resolution:** Added `source_id: str` to `KBRelationship` in Section 4. `write_relationship` signature unchanged (takes a `KBRelationship`) — `source_id` is now part of the model.

---

### ~~MODEL-4: Add confidence scoring to KBRelationship~~ ✓ REVIEWED
**Source:** Domain SME (Important)
**Evidence:** KBNode carries confidence but KBRelationship does not; MITIGATES/CONFLICTS/DEPENDS_ON are high-stakes assertions
**Resolution:** Ignored. Relationship confidence would not come from the same scoring pipeline as nodes — the resolution LLM call has no logprobs/verification agent wiring. Adding `status` to relationships would also complicate the review UI. Heuristic validation already drops malformed relationships. Deferred indefinitely.

---

### ~~REVIEW-1: Add SLA and default resolution to PENDING_REVIEW~~ ✓ REVIEWED
**Source:** SW Engineer (Important), Data Engineer (Important), Domain SME (Important), UX Designer (Important)
**Evidence:** Section 8 Review Flow — no timeout or escalation for `AWAITING_REVIEW` state
**Resolution:** Added `review_timeout_hours: int = 72` to `ExtractionConfig`. Section 8 documents the contract: on expiry, pending nodes default to `REJECTED`, job transitions to `WRITING` with approved nodes only, timeout event logged in MLflow. Enforcement stubbed as `async def enforce_review_timeout(job_id) -> None: ...` — requires durable queue (ARQ/Redis) to fire reliably; not implemented for MVP asyncio queue.
**Post-review update (2026-04-24):** Reverted. `review_timeout_hours` field, the SLA paragraph, and the enforcement stub all removed — three artefacts with zero runtime effect on MVP infra is worse than admitting the feature isn't here. Replaced with a one-sentence note in Section 8 Review Flow and a row in Decisions Deferred to v2. Ships together with the ARQ/Redis queue swap.

---

### ~~REVIEW-2: Add correction endpoint for post-approval nodes~~ ✓ REVIEWED
**Source:** Domain SME (Important)
**Evidence:** Section 8 Review Flow — no mechanism to flag or correct a node after it has been approved and written to KB
**Resolution:** Full behaviour spec added to Section 8 under "Node Correction". Accepts `KBNodeEdit` payload (`title`, `description`). `operator` role only. On success: KB store updated, vector embedding regenerated and upserted, `NodeMetadata.last_edited_by` + `last_edited_at` set, MLflow event logged with before/after values. UI should surface corrected nodes distinctly.
**Post-review update (2026-04-24):** Reverted. `PATCH /graph/{node_id}` endpoint, the Node Correction subsection, and `NodeMetadata.last_edited_by` / `last_edited_at` all removed. MVP reviewers edit at approval time via `ApproveRequest.decisions[].edited_content` (the `KBNodeEdit` model is retained for that path); already-approved nodes on the Markdown KB can be corrected by manual markdown edit + re-embed. The endpoint earns its keep once the KB moves to Notion/Neo4j — deferred to v2. See new row in Decisions Deferred.

---

### ~~REVIEW-3: Add bulk approval to review flow~~ ✓ REVIEWED
**Source:** UX Designer (Important), Domain SME (implied)
**Evidence:** Section 8 Review Flow — only per-node decisions described; one-hour meetings may produce many nodes
**Resolution:** `POST /jobs/{id}/approve` now accepts `ApproveRequest(bulk_action, decisions)`. `BulkAction` approves all nodes above a threshold with optional `exclude` list. `decisions` list provides per-node overrides (approve/reject/edit/reason) and runs after `bulk_action`. Review provenance (`approved_by`, `approved_at`, `approval_method`) added to `NodeMetadata`; `auto_mode` jobs set `approval_method="auto"`.
**Post-review update (2026-04-24):** Two simplifications. `BulkAction(action: Literal["approve_above_threshold"], threshold, exclude)` collapsed to `BulkApproveRule(threshold, exclude)` under a named `ApproveRequest.approve_above_threshold` field — the single-member Literal discriminator was modelling an option that isn't there. `approval_method: Literal["individual", "bulk", "auto"]` promoted to `ApprovalMethod` StrEnum for consistency with `NodeStatus` / `NodeState` / `JobStatus`.

---

### ~~UX-1: Define sub-stage progress contract for GET /jobs/{id}~~ ✓ REVIEWED
**Source:** UX Designer (Critical)
**Evidence:** Section 8 — "per-stage progress" mentioned but undefined
**Resolution:** Added `JobResponse` model to Section 8: `{job_id, status, current_stage, stage_progress: {message}, elapsed_seconds, error}`. Dropped `pct` and `estimated_remaining_seconds` — only transcription has provider-reported progress and time estimates would be unreliable. `message` carries meaningful per-stage info (e.g. "Extracting: 3/4 agents complete"). SSE upgrade path noted for v2.
**Post-review update (2026-04-24):** `StageProgress(message: str)` single-field wrapper inlined — `JobResponse.stage_progress` is now `str | None` directly. No other fields were planned (`pct` and `eta` explicitly dropped), so the wrapper added nothing.

---

### ~~UX-2: Define FAILED error payload shape~~ ✓ REVIEWED
**Source:** UX Designer (Critical) — see also FAILED-1
**Evidence:** Section 8 Job Status Model — FAILED has no payload
**Resolution:** `JobResponse.error: ErrorPayload | None` (added in UX-1) exposes the full error payload in `GET /jobs/{id}` when `status=FAILED`. `ErrorPayload.tokens_used` replaced with `usage: dict[str, list[UsageRecord]]` — `UsageRecord(call_type, units)` with `CallType` enum (`LLM_INPUT`, `LLM_OUTPUT`, `EMBEDDING`, `TRANSCRIPTION`) unifies token and audio-second tracking under one field.
**Post-review update (2026-04-24):** See FAILED-1 post-review note — `job_id` removed, `usage` retyped to `dict[JobStatus, ...]` for type safety.

---

### ~~UX-3: Add notification mechanism for AWAITING_REVIEW~~ ✓ REVIEWED
**Source:** UX Designer (Important)
**Evidence:** Section 8 — no webhook, SSE, or notification on state change; users must actively poll
**Resolution:** Added `callback_url: str | None` to `TranscriptDocument` and a "Job Callbacks" subsection to Section 8. API POSTs `{job_id, status, timestamp}` on `AWAITING_REVIEW` and `DONE`. Fire-and-forget, no retries for MVP. Noted that Streamlit cannot receive inbound POSTs — polling remains the Streamlit model. SSE deferred to v2.
**Post-review update (2026-04-24):** Reverted. `callback_url` field, `JobCallback` model, and the Job Callbacks subsection all removed. MVP has no inbound-HTTP consumer — Streamlit polls `GET /jobs/{id}` and no external consumer (CI, Teams bot, etc.) exists in the thesis scope. Webhook fan-out + SSE stream folded into a single row in Decisions Deferred to v2.

---

### ~~OBSERVABILITY-1: Add prompt/response content to MLflow audit log~~ ✓ REVIEWED
**Source:** Security Expert (Important)
**Evidence:** Section 9 — MLflow captures metadata only; prompt and response content not mentioned
**Resolution:** Added note to Section 9: `mlflow.langchain.autolog()` already captures prompt/response content as artifacts automatically. Flagged as sensitive data. Access-controlled separation from operational metrics noted as v2 hardening — not enforced for MVP (local, single-user).

---

### ~~OBSERVABILITY-2: Add MLflow deep link to Streamlit~~ ✓ REVIEWED
**Source:** UX Designer (Minor)
**Evidence:** Section 9 Streamlit integration — "View in MLflow" button with no deep link
**Resolution:** `mlflow_run_id: str | None` added to `JobResponse`. Section 9 Streamlit integration updated: button deep-links to `http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}` using `mlflow_run_id` from the job response.
**Post-review update (2026-04-24):** `experiment_id` is no longer exposed on the API. New `ObservabilityConfig(mlflow_tracking_uri, mlflow_experiment_name)` added to `SeshatConfig`. At startup the API resolves `experiment_name → experiment_id` via the MLflow client and caches it in-process; the Streamlit deep link uses the cached value plus `mlflow_run_id` from `JobResponse`. Keeps an MLflow internal ID off the public API shape.

---

### ~~CONFIG-1: Nested ExtractionConfig/RAGConfig should be BaseModel not BaseSettings~~ ✓ REVIEWED
**Source:** SW Engineer (Important — downgraded to Minor by DA)
**Evidence:** Section 3 — nested config blocks inherit `BaseSettings`, which can create dual env resolution paths
**Resolution:** All nested configs (`ExtractionConfig`, `RAGConfig`, `SecretsConfig`) changed to `BaseModel`. Root renamed `Settings` → `SeshatSettings(BaseSettings)`, override renamed to `SeshatConfigOverride`. Pattern documented in Section 3 note.

---

### ~~CONFIG-2: Document get_request_settings merge semantics explicitly~~ ✓ REVIEWED
**Source:** SW Engineer (Critical → Important per DA)
**Evidence:** Section 3 Per-Request Overrides — `model_copy(update=...)` replaces nested objects wholesale; partial nested overrides silently lose unset fields
**Resolution:** Implemented deep-merge in `get_request_settings`: nested `BaseModel` fields are merged field-by-field using `exclude_unset` — only explicitly provided override fields replace base values. Also renamed `SeshatSettings` → `SeshatConfig` and `SeshatSettingsOverride` → `SeshatConfigOverride` for naming consistency with all other `*Config` classes.
**Post-review update (2026-04-24):** The shipped implementation was only one level deep — a nested override like `extraction.verification.model` still wholesale-replaced `extraction.verification`, which is the exact bug CONFIG-2 was meant to prevent. Body replaced with `...` and a contract docstring that spells out the recursion-to-any-depth, immutability, and shallow-copy pitfall requirements. Actual implementation deferred to the implementation phase.

---

### ~~INFRA-1: Add worker redundancy and restart policy~~ ✓ REVIEWED
**Source:** Solutions Architect (Critical)
**Evidence:** Section 10 — single `worker` service, no restart policy or replica count
**Resolution:** `restart: unless-stopped` added to `worker` in Section 10. v2 note added: multiple replicas require durable queue (ARQ/Redis) — with MVP asyncio queue, replicas would race on the same in-memory task list.

---

### ~~INFRA-2: Add deployment and rollback strategy note~~ ✓ REVIEWED
**Source:** Solutions Architect (Important)
**Evidence:** Section 10 — no CI/CD, versioning, or rollback strategy mentioned
**Resolution:** Added "Deployment and Rollback" subsection to Section 10. Image versioning: git commit SHA tags. In-flight jobs: asyncio queue is in-memory — wait for active jobs to finish before deploying; `AWAITING_REVIEW` jobs are safe to deploy through. Rollback: redeploy previous SHA tag; KB and MLflow data on named volumes are unaffected.

---

### ~~DATA-1: Define TranscriptDocument.id generation strategy~~ ✓ REVIEWED
**Source:** Data Engineer (Minor)
**Evidence:** Section 2 TranscriptDocument — `id: str` with no generation strategy
**Resolution:** `id: str  # UUID4, generated at job creation time` already present in `TranscriptDocument` in Section 2. The same UUID4 is used as `job_id` in `NodeMetadata` and as the blob storage path key.

---

### ~~SECRETS-1: Simplify SecretsProvider to ENV + AWS only for MVP~~ ✓ REVIEWED
**Source:** User (explicit decision)
**Evidence:** Section 7 SecretsConfig — `SecretsProvider` includes `ENV`, `AWS`, `AZURE`, `VAULT`
**Resolution:** `AZURE` and `VAULT` commented out in `SecretsProvider` (v2). `vault_url` and `vault_addr` removed from `SecretsConfig` — only `region` and `secret_path_prefix` remain. Factory pattern and future hardening note updated accordingly.

---

### ~~BLOB-1: Rename S3 references to generic blob storage; make S3 a pluggable implementation~~ ✓ REVIEWED
**Source:** User (explicit decision)
**Evidence:** Section 2 S3 Artifact Storage, Section 10 Docker Compose (`localstack`), Section 7 SecretsConfig — S3 is referenced directly throughout the spec as a concrete service rather than an abstraction
**Resolution:** Section 2 renamed to "Blob Storage"; pipeline references updated. `AbstractBlobStore` (`put`, `get`, `exists`) added to Section 6 factory diagram and interfaces. `S3BlobStore` is the sole MVP implementation. `BlobStoreProvider` StrEnum and `BlobStoreConfig` (`bucket`, `region`, `endpoint_url`) added to Section 3; `blob_store` field added to `SeshatConfig`. Section 10 localstack comment updated. Bucket structure remains an `S3BlobStore` implementation detail.

---

### ~~MLFLOW-1: Add MLflow ↔ LLM framework wiring note~~ ✓ REVIEWED
**Source:** User (explicit requirement)
**Evidence:** Section 9 — MLflow 3 described but wiring to LLM framework not mentioned; TBD table already updated
**Resolution:** Section 9 already states: "LangChain is the LLM orchestration framework. `mlflow.langchain.autolog()` instruments all agent calls automatically — no manual trace wiring required." TBD table also resolved with the same pattern.

---

## Summary Table

| ID | Tier | Severity | Area |
|----|------|----------|------|
| AUTH-1 | 1 | Critical | Security |
| AUTH-2 | 1 | Critical | Security |
| SECURITY-1 | 1 | Critical | Security |
| SECURITY-2 | 1 | Important | Security |
| CHUNK-1 | 1 | Critical | Pipeline |
| CHUNK-2 | 1 | Important | RAG/Pipeline |
| COST-1 | 1 | Critical | FinOps |
| COST-2 | 1 | Critical | FinOps |
| COST-3 | 1 | Important | FinOps |
| AUTO-1 | 1 | Critical | Security/Governance |
| FAILED-1 | 1 | Critical | API/UX |
| EVAL-1 | 2 | Critical | AI/ML |
| EVAL-2 | 2 | Critical | AI/ML |
| MODEL-1 | 2 | Important | AI/ML |
| QUEUE-1 | 2 | Critical | Infra |
| IDEMPOTENCY-1 | 2 | Critical | Data |
| CONSISTENCY-1 | 2 | Critical | Data |
| SCHEMA-1 | 2 | Critical | Data |
| ASYNC-1 | 2 | Critical | SW Eng |
| RAG-1 | 2 | Important | AI/ML |
| RAG-2 | 2 | Important | RAG |
| CONTEXT-1 | 2 | Important | AI/ML |
| MODEL-2 | 3 | Important | Data Model |
| MODEL-3 | 3 | Important | Data Model |
| MODEL-4 | 3 | Important | Data Model |
| REVIEW-1 | 3 | Important | API |
| REVIEW-2 | 3 | Important | API |
| REVIEW-3 | 3 | Important | UX |
| UX-1 | 3 | Important | UX |
| UX-2 | 3 | Important | UX |
| UX-3 | 3 | Important | API/UX |
| OBSERVABILITY-1 | 3 | Important | Security/Observability |
| OBSERVABILITY-2 | 3 | Minor | UX |
| CONFIG-1 | 3 | Minor | Config |
| CONFIG-2 | 3 | Important | Config |
| INFRA-1 | 3 | Important | Infra |
| INFRA-2 | 3 | Important | Infra |
| DATA-1 | 3 | Minor | Data Model |
| SECRETS-1 | 3 | — | Config/Secrets |
| BLOB-1 | 3 | — | Architecture/Storage |
| MLFLOW-1 | 3 | — | Observability |
