# Seshat Spec — Review TODO List (Second Pass)

**Date:** 2026-04-24
**Spec reviewed:** `docs/superpowers/specs/2026-04-21-seshat-design.md`
**Context:** Second genai-review pass on the Seshat design spec. The spec had already gone through one review round (see `2026-04-21-seshat-review-todos.md`); this pass reviewed the updated spec for self-containedness, internal consistency, and completeness.

**Panel:** Software Engineer, Data Engineer, AI/ML Engineer, Solutions Architect, AI System Designer, Security Expert, Domain SME, UX Designer, FinOps — 9 agents (no Legal/Compliance) + Devil's Advocate + Tech Lead synthesis.

**How to resume:** Read this file top-to-bottom. Each item describes the gap, its evidence in the spec, the recommended fix, and its severity. Address Tier 1 first (blocking), then Tier 2 (pre-implementation), then Tier 3 (refinements). DA Conflicts at the bottom require a judgment call before their items can be closed.

---

## Tier 1 — Blocking (must fix before implementation)

All 7 items resolved — 2026-04-24.

---

### ✓ WRITING-1: WRITING-state crash leaves job stranded with no recovery path

**Source:** Solutions Architect (Important), DA (Challenge 9)
**Evidence:** Section 10 — `worker: restart: unless-stopped`; Section 4 — rollback `node_id` list tracked in-memory; Section 8 — idempotency key only handles `FAILED` state, not `WRITING`
**Gap:** Worker restarts automatically after a crash. If the crash happens mid-`WRITING`, the in-memory `node_id` list (used for rollback) is gone. The job is stuck in `WRITING` on restart — not `FAILED` — so the idempotency key re-run path never triggers. Some KB files may have been written; the vector store has no corresponding entries. There is no defined path to transition a stranded `WRITING` job to `FAILED` so the user can retry.
**Fix:** Define a startup recovery procedure: on worker boot, query all jobs in `WRITING` state; transition them to `FAILED` with `recoverable=True` and `reason="Worker restarted during write — KB state unknown; retry will re-run from extraction"`. Since rollback list is gone, also specify: retry of a stranded WRITING job must first scan KB for any nodes with that `job_id` and delete them before re-running. Document this in Section 8 (FAILED State and Recovery) and Section 10 (Deployment).

---

### ✓ FFMPEG-1: ffmpeg execution has no input validation, timeout, or sandboxing

**Source:** Security Expert (Critical), DA (Challenge 5 — adds path traversal via filename)
**Evidence:** Section 2, lines 41–42 — video files passed to ffmpeg via `ffmpeg-python` wrapper; Section 8, lines 769–770 — file received as `multipart/form-data` with no path handling defined
**Gap:** No file magic byte validation before passing to ffmpeg. No subprocess timeout. No sandboxing. DA adds: the uploaded filename is not sanitised — a crafted filename with path traversal characters (e.g. `../../etc/cron.d/seshat`) could escape the temp directory on naive implementation.
**Fix:** Specify in Section 2: (1) validate file magic bytes against allowed containers (mp4, mkv, webm) before invoking ffmpeg and reject others; (2) enforce a subprocess timeout (e.g. 5 minutes); (3) sanitise the uploaded filename — strip path separators and use a system-generated temp filename, not the original. Note ffmpeg supply-chain risk (known CVE history) and document that video intake is the highest-risk attack surface.

---

### ✓ REVIEWER-1: Reviewer workflow is blind — no confidence breakdown, no relationship visibility, no transcript context

**Source:** Domain SME (Critical), UX Designer (Finding 2, 3, 6, 7)
**Evidence:** Section 8, Review Flow — `ApproveRequest` defined but no spec for what the Streamlit UI must display; Section 4 — `KBNode.relationships` and `confidence` stored but not surfaced at review time
**Gap:** Reviewers see a node title and description and must make a binary approve/reject decision. They cannot see: why the node scored low confidence, what the confidence breakdown is (logprobs vs. verification vs. heuristics), the source transcript excerpt inline, what relationships were extracted and to which nodes, or whether the node conflicts with an existing KB decision. For high-stakes ADR approvals this is blind triage.
**Fix:** Add a "Reviewer UI Requirements" subsection to Section 8 (or a dedicated Section for the Streamlit scope) specifying that for each `PENDING_REVIEW` node the UI must surface: (a) `source_quote` inline, (b) confidence score with component breakdown, (c) the node's extracted relationships (read-only, with hyperlink to target node), (d) existing same-type KB nodes that were flagged as SUPERSEDES/AMENDS/CONFLICTS candidates by the resolution step. This does not need to be a full UI spec — it is a data contract between the API and the UI.

---

### ✓ UI-SCOPE-1: Streamlit UI scope entirely undefined

**Source:** UX Designer (Important, DA upgraded to Critical)
**Evidence:** "Streamlit UI" referenced 12+ times throughout the spec but never defined — no screens, no flows, no data bindings
**Gap:** The implementation team has no specification for what the UI must do. Given the reviewer workflow is the primary quality gate for the knowledge base, an undefined UI means an invented UI — and an invented reviewer workflow means unknown correctness guarantees.
**Fix:** Add a Section (or subsection in Section 1 or Section 8) defining the Streamlit UI scope at the screen/flow level: (1) Job submission page (file upload, metadata entry, submit); (2) Job status page (pipeline progress, elapsed time, error + retry); (3) Review page (pending nodes, confidence, source quote, relationships, bulk approve, per-node approve/reject/edit); (4) Graph query page (filters, node preview, relationship view); (5) MLflow deep-link integration. Does not need pixel-level design — functional scope and data requirements are sufficient.

---

### ✓ CONFLICTS-1: CONFLICTS relationship detected but triggers no NodeState transition

**Source:** DA (Challenge 7)
**Evidence:** Section 4, lines 458–462 — `NodeState` enum has `CURRENT`, `AMENDED`, `SUPERSEDED` only; Section 4, lines 541–546 — state transitions triggered only for `SUPERSEDES` and `AMENDS`; Section 5, lines 605–608 — heuristic validation drops malformed rels but takes no semantic action on `CONFLICTS_WITH`
**Gap:** When the resolution agent detects a `CONFLICTS_WITH` relationship, it writes the relationship to the graph but both nodes remain `NodeState.CURRENT`. A reviewer reading the KB has no signal that a currently-active node has a live conflict with another currently-active node unless they manually traverse the graph.
**Fix:** Either (a) define a `NodeState.CONFLICTED` value and set it on both nodes when a `CONFLICTS_WITH` relationship is written, or (b) document explicitly that `CONFLICTS_WITH` is a graph-level annotation only and reviewers are expected to discover conflicts via the graph query UI. If (b), add a UI requirement: `GET /graph/{node_id}` response must surface any `CONFLICTS_WITH` relationships prominently so the review UI can flag them.

---

### ✓ BUDGET-1: $2.50 extraction budget default appears insufficient — worst-case ~$13.76

**Source:** FinOps (Finding 9), DA (Challenge 6 — verified arithmetic)
**Evidence:** Section 3, lines 286–304 — `max_budget_usd=2.50`; prompt budget: 50 chunks × 4 agents × ~12,700 input tokens + 2,048 output tokens; at claude-sonnet-4-6 pricing (~$3/$15 per MTok) worst case ≈ $7.62 input + $6.14 output = $13.76
**Gap:** The default budget cap will fail most real-world jobs on dense transcripts on the first attempt. Users will hit `FAILED` with `recoverable=True`, need to know to raise `max_budget_usd`, restart the config, and retry — making the default not a safety net but a guaranteed first-run failure at scale.
**Fix:** Either (a) revise the default based on measured baselines from the eval corpus (preferred), or (b) document the calculation explicitly in the spec so implementers understand the default is a conservative cap, not an expected cost. Add a note: "The default $2.50 cap is intentionally conservative — it will fail on max-length transcripts. Calibrate after running `seshat eval` on real data." Also specify exactly when the budget check runs (after each agent call? per stage?) — see BUDGET-2.

---

### ✓ BUDGET-2: Budget check timing unspecified

**Source:** FinOps (Finding 2)
**Evidence:** Section 8, FAILED State — "worker converts usage records to cost via price table and compares against cap"; no specification of when this check runs
**Gap:** If the check only runs after all agents complete, a job can massively overshoot the cap before it is enforced. If it runs before each call, the spec needs to say so.
**Fix:** Specify in Section 8 or Section 3: "The budget check runs after each agent call. Before dispatching the next agent call, the worker computes cumulative cost for the current stage and aborts if the next call would exceed `max_budget_usd`."

---

## Tier 2 — Pre-implementation (define before writing code)

All 12 items resolved — 2026-04-24.

---

### ✓ ROLLBACK-1: Rollback node_id tracking must be durable, not in-memory

**Source:** Solutions Architect (Finding 2), DA (Simplicity flag)
**Evidence:** Section 4, lines 544–546 — "worker tracks the node_id's written during a job"; no persistence mechanism specified
**Gap:** In-memory tracking is lost on any worker crash, making the rollback guarantee illusory. The `restart: unless-stopped` policy means crashes will happen.
**Fix:** Specify that the worker persists the rollback state (list of `node_id`s written so far and `state` changes made) to a durable location before making each KB write. The simplest option: append to a `{job_id}/rollback.json` in blob storage before each write, so the file survives a worker restart and can be read by the startup recovery procedure (WRITING-1).

---

### ✓ XPROVIDER-1: Cross-provider verification enforced by convention — add startup validation

**Source:** Software Engineer (Finding 4), Data Engineer (Finding 9), AI/ML Engineer (Finding 8)
**Evidence:** Section 4, line 513 — "enforced by convention at configuration time, not at runtime"
**Gap:** If both `ExtractionConfig.llm.provider` and `ExtractionConfig.verification.provider` are set to the same value, the correlated-error risk that cross-provider verification is designed to prevent is silently present. The system continues to run and produce unreliable confidence scores without warning.
**Fix:** Add a `model_validator` on `ExtractionConfig` that raises a `ValueError` at startup if `verification` is set and `verification.provider == llm.provider`. Document the error message: "Verification provider must differ from extraction provider to avoid correlated confidence scoring."

---

### ✓ SOURCE-QUOTE-1: source_quote not verified as actual transcript substring

**Source:** AI/ML Engineer (Finding 5)
**Evidence:** Section 4, lines 520–523 — output validation checks Pydantic schema conformance only; `source_quote` is described as "exact transcript excerpt" but this is not enforced
**Gap:** An extraction agent can produce a `source_quote` that is not a substring of the original transcript (hallucinated or injected). This field is later re-injected into resolution agent prompts as grounding evidence — a fabricated quote is therefore an undetected prompt injection vector.
**Fix:** After extraction and before resolution, verify that each node's `source_quote` is a substring of `TranscriptDocument.raw_text` (or a normalised version of it). Nodes that fail this check are rejected with `status=REJECTED` and logged. Document this as step 2b in the extraction pipeline.

---

### ✓ RETRY-POLICY-1: No retry policy defined for transient failures

**Source:** Data Engineer (Finding 12)
**Evidence:** Section 8, Recoverable failures — lists transient errors as recoverable but defines no automatic retry logic
**Gap:** As specified, all transient failures (API timeout, rate limit) immediately mark the job `FAILED` with `recoverable=True` and require the user to manually call `POST /jobs/{id}/retry`. For rate limits especially, immediate retry will hit the same limit. There is no backoff.
**Fix:** Specify a per-stage retry policy in Section 8: "Transient errors (API timeout, HTTP 429) are retried up to N times (default: 3) with exponential backoff (base 2s, max 60s, jitter) before the stage transitions to `FAILED`. The retry count and backoff parameters are configurable in `TranscriptionConfig` and `ExtractionConfig`." If automatic retry is intentionally omitted for MVP simplicity, document that explicitly.

---

### ✓ RELATIONSHIP-EVAL-1: Relationship precision/recall not in eval targets

**Source:** Domain SME (Findings 2, 9)
**Evidence:** Section 12, Precision/Recall Targets table — covers `ConceptType` only; no relationship extraction metrics
**Gap:** The pipeline can achieve high node extraction precision while silently misassigning action items to wrong participants or missing ADR conflicts. Neither is caught by the current eval.
**Fix:** Extend the eval target table in Section 12 to include relationship types. At minimum: `ASSIGNED_TO` precision ≥ 0.90 (action items assigned to wrong person is worse than a missed action item); `CONFLICTS_WITH` recall ≥ 0.75 (missed conflicts accumulate silently). The eval corpus format (`raw_text + expected_nodes`) must be extended to include `expected_relationships: list[KBRelationship]`.

---

### ✓ THRESHOLD-1: Single threshold sweep conflates per-type domain trade-offs

**Source:** Domain SME (Finding 3)
**Evidence:** Section 12, Threshold Calibration — single global threshold swept 0.5→0.9; Section 12, Precision/Recall Targets — RISK is recall-biased, ADR is precision-biased
**Gap:** A single global threshold cannot satisfy both a recall-biased (RISK) and a precision-biased (ADR) target simultaneously — one will be sacrificed for the other.
**Fix:** Clarify in Section 12 that the threshold calibration sweeps per `ConceptType` and that the output is either a per-type threshold map or a justified single value with documented trade-offs. `ExtractionConfig.confidence_threshold` could remain a single value for simplicity, but the calibration step should show the per-type trade-off curve so the operator can make an informed choice.

---

### ✓ AMENDS-DEF-1: No operational definition of AMENDS vs SUPERSEDES for resolution agent

**Source:** Domain SME (Finding 1)
**Evidence:** Section 4, RelationshipType comments — `SUPERSEDES: "fully replaces"`, `AMENDS: "partial update or clarification"` — no agent guidance on when to apply each
**Gap:** The resolution agent must choose between these two for every potential cross-meeting relationship. Without criteria, it will guess — and wrong choices collapse or fragment decision history.
**Fix:** Add operational criteria to the resolution agent's system prompt spec (or document here for implementation): `SUPERSEDES` applies when the new node renders the prior decision actionable-irrelevant (the old decision would no longer be followed). `AMENDS` applies when the new node narrows, extends, conditionally qualifies, or corrects a detail of the prior decision while leaving it broadly active. Include 2 labelled examples per type in the eval corpus.

---

### ✓ EMBEDDING-BUDGET-1: Embedding cost unbudgeted; up to 200 calls per job

**Source:** FinOps (Finding 5), DA (Challenge 10)
**Evidence:** Section 8, `CallType.EMBEDDING` exists; Section 5, retrieval loop embeds each new `KBNode` individually; Section 3, `RAGConfig` — no embedding budget cap
**Gap:** Up to 200 embedding calls per job (50 chunks × 4 concept types) in the RAG phase alone, on top of extraction. `CallType.EMBEDDING` is tracked in `UsageRecord` but there is no cap — embedding cost accumulates invisibly against no ceiling.
**Fix:** Add `max_embedding_usd: float | None = None` to `RAGConfig`. Document that embedding cost is tracked in `UsageRecord` per stage and accumulates toward the stage budget. Alternatively, fold embedding into the per-stage `max_budget_usd` and document that the cap covers all LLM + embedding calls for that stage.

---

### ✓ CACHE-OBSERVABILITY-1: Prompt caching cache hit rate untracked

**Source:** FinOps (Finding 3), AI/ML Engineer (Finding 4)
**Evidence:** Section 4, Prompt Caching — caching specified as first-class but no measurement; Section 3, lines 293–304 — retrieved_context (≤4000t) varies per job so OpenAI automatic prefix caching may not cache it
**Gap:** The cost projections assume caching works and saves ~$0.30 per job. If retrieved_context is not cached (because it varies), actual savings are much lower. There is no metric to know.
**Fix:** Add to Section 9 (Observability): cache hit/miss per agent call must be logged to MLflow. For Anthropic: `cache_read_input_tokens` and `cache_creation_input_tokens` from the response headers. For OpenAI: `cached_tokens` in the usage object. Document that the first job after startup will have zero cache hits (cold start) — this is expected and should not trigger alerts.

---

### ✓ MARKDOWN-LOCK-1: MarkdownKBStore concurrent writes — no file locking

**Source:** Data Engineer (Finding 13)
**Evidence:** Section 6, Interfaces — `async def` methods with `asyncio.to_thread()` for blocking I/O; no locking mentioned
**Gap:** Multiple async tasks could attempt concurrent writes to the same markdown file (e.g. parallel agent completions writing nodes for the same job). No file-level lock is specified.
**Fix:** Specify in Section 6 (MVP: Markdown + Git): `MarkdownKBStore` must acquire a per-`node_id` asyncio lock before writing. A module-level `defaultdict(asyncio.Lock)` keyed on `node_id` is sufficient for the single-process MVP. Document that this does not extend to multi-process (multi-replica) scenarios — concurrent replicas require the ARQ/Redis upgrade.

---

### ✓ RETRIEVAL-GATE-1: Retrieval quality baseline not enforced as a release gate

**Source:** Domain SME (Finding 6), AI System Designer (Finding 1 — DA downgraded from Critical)
**Evidence:** Section 5, Retrieval Quality Baseline — "recall@5 >= 0.7 required before MVP ships" stated but no enforcement mechanism
**Gap:** `seshat eval` is a CLI command; there is no CI gate or documented process that prevents processing real meetings before the retrieval baseline is measured.
**Fix:** Add a note to Section 5 and Section 12: "No real meeting recordings may be processed until `seshat eval` has been run and `recall@5 >= 0.7` is confirmed on the retrieval baseline. This is an honour-system gate for MVP — enforce it at the CI level in v2."

---

### ✓ SYNC-SECRETS-1: AbstractSecretsProvider.get_secret() is synchronous

**Source:** Software Engineer (Finding 9), DA (Challenge 3 — proposes Important if called in hot path)
**Evidence:** Section 7, line 711 — `def get_secret(self, key: str) -> str` (sync); Section 6, lines 682–683 — all other blocking I/O must use `asyncio.to_thread()`
**Gap:** `AWSSecretsProvider` will make an HTTP call to AWS Secrets Manager. If this is called inside the asyncio extraction loop (per-agent, not just at startup), it blocks the event loop and serialises all concurrent agent calls.
**Fix:** Either (a) change `get_secret` to `async def` and document that blocking implementations must wrap in `asyncio.to_thread()`, consistent with all other abstract interfaces — or (b) document explicitly that secrets are resolved once at startup (not per-agent invocation) and cached in-process, making the sync interface safe. Pick one and document it in Section 7.

---

## Tier 3 — Refinements (address before first real job run)

All 13 items resolved — 2026-04-24.

---

### ✓ TRUNCATION-ORDERING-1: RAG context truncation drops newest evidence first

**Source:** DA (Challenge 8)
**Evidence:** Section 3, lines 315–322 — assembler truncates at `max_context_tokens`; no ordering defined for assembled context
**Gap:** If recently-created KB nodes are appended last in the assembled context, tail truncation drops the most-recent evidence first — the worst possible behaviour for a system tracking decision evolution over time.
**Fix:** Specify in Section 5 (Retrieval Flow): the context assembler orders retrieved nodes by recency descending (most recent first) before serialisation. Truncation at `max_context_tokens` then drops the oldest evidence, not the newest. Log truncation events with count of dropped nodes.

---

### ✓ WITHIN-MEETING-CONFLICT-1: Within-meeting conflict reversals silently discarded

**Source:** Domain SME (Finding 5), AI/ML Engineer (Finding 10)
**Evidence:** Section 4, lines 524–528 — "merge step keeps the most recent settled position — no SUPERSEDES within a single job"
**Gap:** If a meeting reverses a decision ("use Kafka" → "actually REST"), only the final position is kept. The reversal history is lost. A reviewer cannot tell whether a decision was unanimous or contested.
**Fix:** Document the trade-off explicitly: within-meeting deduplication prioritises a clean final KB over preserving debate history. If the reversal is significant, the transcript `source_quote` on the surviving node should reflect the final position — the earlier quote should not be used. Consider logging the count of within-meeting merges per job to MLflow as a quality signal (a job with many merges may indicate a contentious or poorly-transcribed meeting worth manual review).

---

### ✓ PRECISION-AT-K-1: Precision@5 not measured alongside Recall@5

**Source:** AI System Designer (Finding 3)
**Evidence:** Section 5, Retrieval Quality Baseline — only recall@5 defined
**Gap:** High recall@5 with low precision@5 means resolution agents receive many irrelevant candidates — noisy resolution, not just incomplete retrieval.
**Fix:** Add precision@5 to the retrieval baseline metrics in Section 5. Acceptable threshold: TBD based on eval data (suggest ≥ 0.6 as a starting point). If precision@5 is low while recall@5 is acceptable, this is the signal to invest in reranking.

---

### ✓ EVAL-OOD-1: Eval corpus too small; no adversarial test set

**Source:** AI/ML Engineer (Findings 1, 2)
**Evidence:** Section 12 — 5–10 synthetic transcripts; no OOD or adversarial cases
**Gap:** 5–10 transcripts is too small to detect distribution shift or adversarial gaming of confidence heuristics.
**Fix:** Document a plan to grow the corpus: (1) 5–10 normal transcripts (current), (2) 2–3 adversarial transcripts (confident-sounding nonsense, injected instructions, ambiguous pronouncements) added before first real-data run. OOD testing (non-technical jargon, highly ambiguous transcripts) deferred to v2 when real data is available.

---

### ✓ EMBEDDING-MODEL-1: text-embedding-3-small may mismatch logical relationship detection

**Source:** AI System Designer (Finding 2)
**Evidence:** Section 3, RAGConfig — `embedding_model: str = "text-embedding-3-small"`; Section 5 — embeddings used for SUPERSEDES/MITIGATES relationship retrieval
**Gap:** General-purpose semantic embeddings conflate semantic similarity with logical coupling. Two ADRs about the same topic but independent may score highly similar; a Risk and an ADR with a MITIGATES relationship may score dissimilar. The retrieval step may surface wrong candidates for resolution.
**Fix:** Document this limitation explicitly in Section 5 and flag it as the primary reason the retrieval baseline must be measured before real use. If recall@5 < 0.7, switching embedding model (e.g. a fine-tuned model or domain-specific encoder) is the first tuning lever — add this to the "tune top_k or adjust embedding model" note already in Section 5.

---

### ✓ RATE-LIMIT-1: No rate limiting on job submission

**Source:** Security Expert (Finding 6)
**Evidence:** Section 8, API Layer — no per-user or global job submission rate limit
**Gap:** A valid `submitter` key can flood the job queue, exhaust LLM API quotas, and fill blob storage.
**Fix:** Add `max_jobs_per_user_per_hour: int` (suggest 10) to the job submission endpoint. Return HTTP 429 when exceeded. Log rate-limit violations per `user_id`. Implement as an in-memory counter for MVP.

---

### ✓ BCRYPT-COST-1: Bcrypt cost parameter not specified

**Source:** Security Expert (Finding 5)
**Evidence:** Section 8, Authentication — "Keys stored as bcrypt hashes" with no cost factor
**Fix:** Specify `bcrypt cost=12` (current 2026 recommendation balancing brute-force resistance and latency). Document that comparison uses `bcrypt.checkpw()` for constant-time verification.

---

### ✓ SECRETS-ROTATION-1: Secrets rotation not documented; singleton not reloaded

**Source:** Security Expert (Finding 4)
**Evidence:** Section 7 — `AbstractSecretsProvider`; Section 3 — `SeshatConfig` loaded at startup as singleton
**Gap:** If an LLM API key is compromised and rotated, the running worker continues using the stale key until redeployed.
**Fix:** Document the rotation procedure in Section 7: rotating a secret requires a worker restart to pick up the new value. For MVP this is acceptable — add a note. For v2, implement a TTL-based secret cache in `AbstractSecretsProvider` so rotation takes effect within N minutes without a full restart.

---

### ✓ MLFLOW-SENSITIVE-1: Full prompts + KB context in MLflow with no access controls

**Source:** Security Expert (Finding 1)
**Evidence:** Section 9 — "`mlflow.langchain.autolog()` captures full prompt inputs and model response outputs as MLflow artifacts automatically"
**Fix:** The spec already notes this is sensitive data. Add an explicit note: for MVP (local, single-user), no additional controls are required. For any multi-user or shared deployment, the MLflow artifact store must be separated from operational metrics with role-based access controls before use. This is a v2 hardening item, but the risk must be communicated to anyone who deploys this.

---

### ✓ BLOB-ENCRYPT-1: Blob artifacts not encrypted

**Source:** Security Expert (Finding 11)
**Evidence:** Section 2, Scope note — "SSE-S3 encryption... is out of scope"
**Gap:** Raw transcripts and extracted decisions are stored unencrypted. The scope note frames this as production hardening, but these are sensitive by default.
**Fix:** The scope note is defensible for a local LocalStack MVP. Strengthen it: "If this system is ever connected to real AWS S3 (rather than LocalStack), SSE-KMS encryption and a private bucket policy must be enabled before any real meeting content is stored. This is not optional." Add this as an explicit pre-production checklist item.

---

### ✓ INIT-CHECKPOINT-1: Init pipeline is a single atomic write with no checkpoint

**Source:** Data Engineer (Finding 4)
**Evidence:** Section 2, Init Pipeline — all nodes written after user confirms; no resume path on failure
**Gap:** If init fails mid-write on a large corpus, the entire run must restart.
**Fix:** For MVP, document this limitation explicitly: "`seshat init` is all-or-nothing. A failure mid-write requires re-running the full init. The git-tracked `data/knowledge_base/` directory acts as the rollback — `git checkout data/knowledge_base/` after a failed init restores a clean state." Add a pre-write dry-run option to the `seshat init` spec: `--dry-run` prints the extraction summary without prompting for write confirmation, allowing cost validation before committing.

---

### ✓ MERGE-DEFINITION-1: "Most recent" in within-meeting merge step not precisely defined

**Source:** Data Engineer (Finding 2)
**Evidence:** Section 4, Chunking — "keeps the most recent settled position"; no definition of "most recent"
**Fix:** Define "most recent" as: the extraction from the chunk with the highest index (i.e. latest in the transcript), regardless of confidence. If chunk ordering is ambiguous (parallel agent results), use chunk start-token position as the tiebreaker. Document this in the Chunking subsection.

---

### ✓ SCHEMA-MIGRATION-1: schema_version defined but no migration strategy

**Source:** Data Engineer (Finding 6)
**Evidence:** Section 6, MVP: Markdown + Git — schema version checked on read; incompatible = hard error; no migration defined
**Fix:** Add a note: "For MVP, schema migrations are handled manually: update all affected markdown files in `data/knowledge_base/` via a migration script and bump the `schema_version` field. A migration script template is provided in `scripts/`. No automated migration is implemented for MVP."

---

## Unresolved DA Conflicts (require judgment call)

These findings have competing severity assessments from domain agents and the DA. A human decision is needed before closing.

---

### DA-CONFLICT-1: Write order rollback severity — Critical vs Important

- **Domain finding (Data Engineer):** write_node() idempotency unspecified; dangling relationships possible — Severity: Critical
- **DA challenge:** Rollback path exists (incomplete, not absent) — Severity: Important
- **Why unresolved:** Solutions Architect independently confirmed the rollback list is in-memory and lost on crash (ROLLBACK-1). If ROLLBACK-1 is fixed (durable rollback log), the Data Engineer's Critical may resolve to Important. Address ROLLBACK-1 first, then reassess.

---

### DA-CONFLICT-2: recall@5 gate severity — Critical vs Important

- **Domain finding (AI System Designer):** no enforcement gate — Severity: Critical
- **DA challenge:** Spec says "tune top_k" — plan exists, enforcement is the gap — Severity: Important
- **Note:** DA is correct that the plan exists. The gap is enforcement — RETRIEVAL-GATE-1 addresses this. Once RETRIEVAL-GATE-1 is applied, this likely resolves to Important.

---

### DA-CONFLICT-3: sync AbstractSecretsProvider severity — Minor vs Important

- **Domain finding (SE):** sync/async inconsistency — Severity: Minor
- **DA challenge:** Blocking AWS SM call in asyncio loop would serialise all concurrent agent calls — Severity: Important
- **Decision needed:** Is `get_secret()` called at startup only (Minor) or per-agent invocation (Important)? Fix SYNC-SECRETS-1 by documenting the call frequency — this conflict resolves automatically.

---

### DA-CONFLICT-4: Streamlit UI scope severity — Important vs Critical

- **Domain finding (UX Designer):** UI scope undefined — Severity: Important
- **DA challenge + Domain SME confirmation:** Primary reviewer workflow entirely uninvented; reviewers are blind without it — Severity: Critical
- **Recommendation:** Treat as Critical. The reviewer UI is the primary correctness gate for the knowledge base. Without it, REVIEWER-1 and UI-SCOPE-1 cannot be addressed. The DA's upgrade is well-supported.

---

### DA-CONFLICT-5: $2.50 budget default severity — Minor vs Important

- **Domain finding (FinOps):** insufficient — Severity: Minor (pre-DA)
- **DA challenge:** verified worst-case ~$13.76 — Severity: Important
- **Recommendation:** Treat as Important. DA's arithmetic is uncontested. Address via BUDGET-1.

---

### DA-CONFLICT-6: RAG context truncation ordering

- **DA challenge:** newest nodes appended last → tail truncation drops most-recent evidence first
- **Why unresolved:** The spec doesn't define retrieval ordering. If the assembler reorders by recency before truncating, the concern dissolves.
- **Recommendation:** Apply TRUNCATION-ORDERING-1 to define the ordering explicitly, which resolves the conflict either way.

---

## Summary Table

| ID | Tier | Severity | Area |
|----|------|----------|------|
| ✓ WRITING-1 | 1 | Critical | Reliability |
| ✓ FFMPEG-1 | 1 | Critical | Security |
| ✓ REVIEWER-1 | 1 | Critical | UX/Correctness |
| ✓ UI-SCOPE-1 | 1 | Critical | UX |
| ✓ CONFLICTS-1 | 1 | Important | Data Model |
| ✓ BUDGET-1 | 1 | Important | FinOps |
| ✓ BUDGET-2 | 1 | Important | FinOps |
| ✓ ROLLBACK-1 | 2 | Important | Reliability |
| ✓ XPROVIDER-1 | 2 | Important | AI/ML |
| ✓ SOURCE-QUOTE-1 | 2 | Important | Security/AI |
| ✓ RETRY-POLICY-1 | 2 | Important | Reliability |
| ✓ RELATIONSHIP-EVAL-1 | 2 | Important | Evaluation |
| ✓ THRESHOLD-1 | 2 | Important | Evaluation |
| ✓ AMENDS-DEF-1 | 2 | Important | AI/Domain |
| ✓ EMBEDDING-BUDGET-1 | 2 | Important | FinOps |
| ✓ CACHE-OBSERVABILITY-1 | 2 | Important | Observability |
| ✓ MARKDOWN-LOCK-1 | 2 | Important | Reliability |
| ✓ RETRIEVAL-GATE-1 | 2 | Important | Process |
| ✓ SYNC-SECRETS-1 | 2 | Minor/Important* | Architecture |
| ✓ TRUNCATION-ORDERING-1 | 3 | Important | AI/RAG |
| ✓ WITHIN-MEETING-CONFLICT-1 | 3 | Minor | Domain |
| ✓ PRECISION-AT-K-1 | 3 | Minor | Evaluation |
| ✓ EVAL-OOD-1 | 3 | Minor | Evaluation |
| ✓ EMBEDDING-MODEL-1 | 3 | Minor | AI/RAG |
| ✓ RATE-LIMIT-1 | 3 | Important | Security |
| ✓ BCRYPT-COST-1 | 3 | Minor | Security |
| ✓ SECRETS-ROTATION-1 | 3 | Minor | Security |
| ✓ MLFLOW-SENSITIVE-1 | 3 | Minor | Security |
| ✓ BLOB-ENCRYPT-1 | 3 | Minor | Security |
| ✓ INIT-CHECKPOINT-1 | 3 | Minor | Reliability |
| ✓ MERGE-DEFINITION-1 | 3 | Minor | Correctness |
| ✓ SCHEMA-MIGRATION-1 | 3 | Minor | Data |

*SYNC-SECRETS-1 severity depends on call frequency — see DA Conflict 3.
