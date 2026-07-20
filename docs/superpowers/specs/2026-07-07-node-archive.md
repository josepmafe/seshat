# Node Archive / Restore — Design Spec

**Date:** 2026-07-07
**Status:** Deferred — design complete; implementation still pending. Refreshed 2026-07-20 against the post-restructure codebase (module paths, the now-implemented vector-store state sync, and the `SearchEngine` retrieval path).

## Scope

Soft-delete for approved KB nodes via a new `NodeStatus.ARCHIVED` value. Archived nodes are excluded from search and browse by default, their VS embeddings are removed, and their outbound state transitions are reversed (same logic as hard delete). Restore is the exact inverse. No schema migration is required.

---

## 1. Data Model

### `NodeStatus` enum

`NodeStatus` gains a fourth value:

```python
class NodeStatus(StrEnum):
    APPROVED = auto()
    PENDING_REVIEW = auto()
    REJECTED = auto()
    ARCHIVED = auto()
```

No Alembic migration is needed — the column is already `TEXT`; the new value is a new string.

### `NodeFilter`

`NodeFilter` in `core/models/api_graph.py` gains:

```python
include_archived: bool = Field(
    default=False,
    description="When True, include ARCHIVED nodes in query and search results.",
)
```

When `include_archived=False` (default), both KB and VS queries exclude archived nodes. When `True`, archived nodes are surfaced alongside all other statuses. If the caller also sets an explicit `status=` filter, that filter takes full precedence: `status=archived` returns only archived nodes; any other `status=` value excludes archived nodes regardless of `include_archived`.

---

## 2. Filtering

### KB (`pg_store.query`)

`PostgresKBStore.query` (`pg_store.py:413`) appends a conditional clause:

```sql
AND status != 'archived'   -- only when include_archived=False and status filter not already set
```

If the caller passes an explicit `status=` filter, that takes precedence and the `include_archived` flag has no further effect on the query.

### VS (`pgvector_store` search)

No VS filter changes are needed. Archived nodes have no VS embedding (deleted by `archive_node`, not re-inserted until `restore_node`). Absent embeddings cannot appear in search results. The embedding lifecycle is the enforcement mechanism — no metadata filtering required.

### `get_neighbours` SQL

`PostgresKBStore.get_neighbours` (`pg_store.py:375`) currently has no status filter — it returns all adjacent nodes including archived ones. Archived nodes retain their relationship rows (archive does not delete relationships), so without a filter they appear in:

- `GET /graph/{node_id}/neighbours` (via `get_node_neighbours`, `graph.py:102`)
- `GET /graph/{node_id}/detail` (via `get_node_detail`, `graph.py:106`)
- `GET /graph/{node_id}/impact` (via `traverse_impact`, `graph.py:114` → `_fetch_neighbours`, `graph.py:308` — note: `traverse_impact` calls `_fetch_neighbours` directly, bypassing even the `_both_current` state check)
- RAG pipeline neighbour expansion in `NodeRetriever._expand_with_neighbours` (`node_retriever.py:115`)

Fix: add `AND n.status != 'archived'` to the `WHERE` clause in `get_neighbours`. One change covers all callers.

Additionally, `_both_current` (`graph.py:349`) checks `node.state == NodeState.CURRENT` but not `node.status`. An archived node can have `state=CURRENT`. Update to:

```python
def _both_current(source: KBNode, target: KBNode) -> bool:
    return (
        source.state == NodeState.CURRENT and source.status != NodeStatus.ARCHIVED
        and target.state == NodeState.CURRENT and target.status != NodeStatus.ARCHIVED
    )
```

---

## 3. `count_inbound_relationships` extension

The existing method (`pg_store.py:223`, current signature `count_inbound_relationships(self, node_id, rel_types=None)`) is extended with an optional `exclude_source_status` parameter **and** an optional `conn`:

```python
async def count_inbound_relationships(
    self,
    node_id: str,
    rel_types: list[str] | None = None,
    exclude_source_status: str | None = None,
    *,
    conn: _Conn | None = None,
) -> int:
```

`conn` is required for correctness, not convenience: `archive_node` sets `status='archived'` and then counts remaining non-archived sources *within the same transaction* (see §4). The count must run on that connection to see the uncommitted status change — otherwise it still counts the node being archived. The method currently executes on `self.pool`; it must switch to `executor = conn or self.pool`, matching `update_node_state` (`pg_store.py:170`) and the other transaction-aware primitives.

The base query remains:

```sql
SELECT COUNT(*) FROM knowledge_base.kb_relationships WHERE target_id=$1
```

Filters are appended conditionally:

- When `rel_types` is set: `AND rel_type = ANY($n::text[])`
- When `exclude_source_status` is set: a JOIN is added rather than a subquery:

```sql
JOIN knowledge_base.kb_nodes n ON n.node_id = source_id
-- plus:
AND n.status != $n
```

Using a JOIN on the PK lets the planner use an index lookup; `NOT IN (subquery)` is avoided.

`NodeRepository.count_inbound_relationships` (`node_repository.py:195`) mirrors the new signature and passes `exclude_source_status` (and `conn`) through.

**Note — this is a different check from hard delete.** `delete_node` reverts state via `count_remaining_state_transition_sources(target_id, excluding_source_id=...)` (`pg_store.py:327`), which excludes exactly one source by ID because the delete *removes* the edge rows. Archive keeps the edge rows and instead excludes sources by *status*, so it cannot reuse that method — excluding a single ID would still count any *other* already-archived source and wrongly block the revert. Hence the `exclude_source_status` variant.

---

## 4. `NodeRepository` — archive and restore

### `archive_node(node_id: UUID) -> KBNode`

1. Load node via `get_node`; raise `NodeNotFoundError` if absent.
2. Assert `node.status == NodeStatus.APPROVED`; raise `NodePreconditionError` if not.
3. Fetch outbound state-transition targets via `get_outbound_state_transition_targets` (`pg_store.py:315`).
4. Open a single DB transaction (`async with self._kb.transaction() as conn:`):
   - Set `status = 'archived'` on the node via `update_node_status(node_id, ARCHIVED, conn=conn)`.
   - For each target, call `count_inbound_relationships(target_id, rel_types=[SUPERSEDES, AMENDS], exclude_source_status='archived', conn=conn)` — the `conn` is what lets the count see the status change just written above. If the count drops to 0, revert the target via `_transition_node_state(target_id, NodeState.CURRENT, conn=conn)`.
5. Outside the transaction: `vs.delete(str(node_id))`.
6. Return the updated node (status overridden in-memory; no extra DB read).

Reverting through `_transition_node_state` (`node_repository.py:97`) — not the raw KB `update_node_state` — is required: it writes the KB `state` *and* patches the VS `state` metadata in lockstep, keeping the two stores in sync (the vector-store state-sync behaviour, `2026-07-12-vector-store-state-sync.md`, is now implemented). The reverted target still has a live embedding, so a stale `state` there would leak it back into `state=CURRENT` retrieval.

### `restore_node(node_id: UUID) -> KBNode`

1. Load node via `get_node`; raise `NodeNotFoundError` if absent.
2. Assert `node.status == NodeStatus.ARCHIVED`; raise `NodePreconditionError` if not.
3. Fetch outbound `SUPERSEDES`/`AMENDS` relationships for the node via the new `get_outbound_state_transition_relationships` (see below).
4. Open a single DB transaction:
   - Set `status = 'approved'` on the node via `update_node_status(node_id, APPROVED, conn=conn)`.
   - For each outbound state-transition relationship, re-apply the forward transition via `_transition_node_state(target_id, SUPERSEDED | AMENDED, conn=conn)` (map `rel_type` through `_STATE_TRANSITIONS`, `node_repository.py:17`).
5. Outside the transaction: re-insert the embedding. **The metadata must go through `_get_vector_store_metadata(node)` (`node_repository.py:271`), not a bare `node.metadata.model_dump(mode="json")`.** That helper appends `{"state": node.state}`, and retrieval now filters on the VS `state` key (`pgvector_store.py`, `_build_semantic_filter` / `_apply_sparse_filter`). Upserting without it produces a stateless embedding that either never matches a `state=CURRENT` query or, worse, resurfaces a superseded node as current:

   ```python
   await self._vs.upsert(str(node_id), node.vector_store_text, _get_vector_store_metadata(node))
   ```

6. Return the updated node.

The `write_node` / `update_node` paths already upsert via `_get_vector_store_metadata` (`node_repository.py:47`, `:64`); `restore_node` must use the same helper for consistency rather than reconstructing the metadata inline.

**New KB store primitive required for restore:** `get_outbound_state_transition_targets` (`pg_store.py:315`) returns only `target_id` strings. `restore_node` needs to know which `rel_type` applies to each target to pick the correct forward state (`SUPERSEDES → SUPERSEDED`, `AMENDS → AMENDED`). A new method is needed:

```python
async def get_outbound_state_transition_relationships(
    self, node_id: str
) -> list[tuple[str, RelationshipType]]:
    """Return (target_id, rel_type) pairs for outbound SUPERSEDES/AMENDS relationships."""
```

`archive_node` continues to use `get_outbound_state_transition_targets` (only needs IDs). `restore_node` uses the new method.

Both methods also require a new `update_node_status` KB store primitive (analogous to `update_node_state`) to set `status` on a node within a transaction:

```python
async def update_node_status(
    self, node_id: str, new_status: NodeStatus, *, conn: _Conn | None = None
) -> None:
    ...
    await conn_or_pool.execute(
        f"UPDATE {self._schema}.kb_nodes SET status=$1 WHERE node_id=$2",
        new_status.value, node_id,
    )
```

---

## 5. `GraphService` — service layer

Two new thin methods:

```python
async def archive(self, node_id: UUID) -> KBNode:
    return await self._repo.archive_node(node_id)

async def restore(self, node_id: UUID) -> KBNode:
    return await self._repo.restore_node(node_id)
```

No new exception types. `NodeNotFoundError` and `NodePreconditionError` propagate naturally from the repository layer and are already handled in the router.

**Guard on `override`:** `GraphService.override` (`graph.py:213`) only checks `approval_method`, not `status`. An admin calling `PUT /graph/nodes/{archived_id}/override` would call `update_node`, which unconditionally calls `vs.upsert`, silently re-inserting the archived embedding. Add a status check at the top of `override`:

```python
if node.status == NodeStatus.ARCHIVED:
    raise NodePreconditionError("Cannot override an archived node — restore it first")
```

---

## 6. API Endpoints

Two new `PATCH` endpoints on the `_nodes` sub-router (`/graph/nodes`), operator-gated:

### `PATCH /graph/nodes/{node_id}/archive`

| Status | Meaning |
|--------|---------|
| 200 | Node archived; returns updated `KBNode` |
| 401 | Missing or invalid API key |
| 403 | Operator role required |
| 404 | Node not found |
| 409 | Node not in `APPROVED` status |

### `PATCH /graph/nodes/{node_id}/restore`

| Status | Meaning |
|--------|---------|
| 200 | Node restored; returns updated `KBNode` |
| 401 | Missing or invalid API key |
| 403 | Operator role required |
| 404 | Node not found |
| 409 | Node not in `ARCHIVED` status |

---

## 7. Relationship with deferred `NodeState.ORPHANED`

The reversion step in `archive_node` (and the existing `delete_node`) currently reverts targets to `NodeState.CURRENT` when their last active superseding source is removed. The deferred `ORPHANED` state would change this to `NodeState.ORPHANED` instead — signalling that the node became current by absence of evidence rather than by active review.

When `ORPHANED` is implemented, only the reversion target in `archive_node` and `delete_node` changes (`CURRENT` → `ORPHANED`); the surrounding logic is identical. The archive feature is designed so this upgrade is a one-line change.

---

## 8. Force Re-ingest Integration

`JobService._delete_job_nodes` (`job.py:395`) currently hard-deletes all prior-job nodes regardless of status. Once archive is available, it should split by status:

- `PENDING_REVIEW` / `REJECTED` → `node_repo.delete_node(cascade=True)` — no audit value, no VS embedding to preserve.
- `APPROVED` → `node_repo.archive_node()` — preserves the KB record, relationships, and audit trail. The state-reversion logic in `archive_node` fires correctly, giving the new job's resolution pass a clean slate.

`JobSubmissionRequest.force` docstring is updated to reflect the new behaviour (currently incorrectly states only PENDING/REJECTED are deleted).

The `force` gate remains admin-only in the router — no role change.

**Edge case — repeated force re-ingest:** `_delete_job_nodes` queries `NodeFilter(job_id=job_id)`. Once `include_archived` defaults to `False`, nodes archived by a *previous* force re-ingest of the same job will not be found, accumulating as orphaned archived nodes in the DB. Fix: use `NodeFilter(job_id=job_id, include_archived=True)` in `_delete_job_nodes` so all statuses are returned before the APPROVED/non-APPROVED split.

---

## 9. Resolution Pipeline Exclusion

Archived nodes must not appear as candidates in any resolution or identification pass.

**VS retrieval (covered automatically):** `NodeRetriever.retrieve` (`node_retriever.py:44`) now searches through the injected `SearchEngine` (`self._search_engine.search`, `node_retriever.py:58`) rather than calling the vector store directly. Either way the guarantee holds: since `archive_node` deletes the VS embedding, archived nodes are absent from search results without any filter change. (The earlier design referenced a "delete-on-archival — TODO" comment at `node_retriever.py:52`; that comment no longer exists after the retrieval refactor — the embedding-deletion mechanism supersedes it.)

**KB hint query (explicit fix required):** `ExtractionOrchestrator._fetch_kb_hints` (`orchestrator.py:211`) calls `paginated_query(NodeFilter(node_type=..., limit=...))` with no status filter. Archived nodes could appear here as identification context hints. Fix: pass `include_archived=False` explicitly (it is already the default on `NodeFilter`, but making it explicit documents the intent).

No changes are needed to the resolution source query (`run_resolution`, `orchestrator.py:152`) — it already filters `status=NodeStatus.APPROVED`, which excludes archived nodes.

**`delete_relationship` reversion must exclude archived sources:** `NodeRepository.delete_relationship` (`node_repository.py:229`) calls `count_inbound_relationships` (`node_repository.py:233`) to decide whether to revert a target's state. This count currently includes archived source nodes — an archived superseding node would prevent the target from reverting to `CURRENT` after its last *active* superseding source is deleted. Fix: pass `exclude_source_status='archived'` in the `delete_relationship` reversion count (same parameter added in §3), threading the transaction `conn` so the count and the delete stay consistent.

**Manual resolution (`POST /graph/nodes/resolve`) is also safe:** `resolve_by_ids` (`graph.py:229`) checks `status == APPROVED` on every supplied node ID before entering the resolution pass, so archived nodes are rejected with a `NodePreconditionError` at the service layer. Target retrieval goes through the VS (same as pipeline), so the embedding-deletion guarantee applies here too.

---

## 10. Testing

**Unit — `tests/unit/app/services/test_job_service.py`** (extend existing):
- `_delete_job_nodes` archives approved nodes and hard-deletes pending/rejected nodes
- `JobSubmissionRequest.force` docstring reflects actual behaviour

**Unit — `tests/unit/app/platform/api/test_graph_router.py`:**
- `TestArchiveNode`: requires auth, 404 on missing, 409 on non-approved, 200 returns node
- `TestRestoreNode`: requires auth, 404 on missing, 409 on non-archived, 200 returns node

**Unit — `tests/unit/app/services/test_graph_service.py`:**
- `TestArchive`: delegates to repo, propagates errors
- `TestRestore`: delegates to repo, propagates errors
- `override` raises `NodePreconditionError` for archived nodes

**Unit — `tests/unit/app/repositories/test_node_repository.py`** (extend existing):
- Archive: state reversion fires when last active source removed; does not fire when other active sources remain; VS `delete` called
- Archive: reverted target's VS `state` metadata is patched to `current` (via `_transition_node_state`), not left stale
- Restore: forward state transition re-applied; VS `upsert` called with metadata that includes the `state` key
- `delete_relationship` reversion excludes archived sources when counting inbound edges

**Unit — `tests/unit/infra/knowledge_store/test_postgres_store.py`** (extend existing — note this is the KB store; `tests/unit/infra/ops_store/test_pg_store.py` is a *different*, unrelated store):
- `count_inbound_relationships` with `exclude_source_status` excludes archived sources correctly
- `count_inbound_relationships` with `conn` sees an uncommitted `status='archived'` written earlier in the same transaction
- `get_neighbours` does not return archived nodes
- `get_outbound_state_transition_relationships` returns correct `(target_id, rel_type)` pairs
