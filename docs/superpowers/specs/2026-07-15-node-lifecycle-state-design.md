# Node Lifecycle State — Design Spec

**Date:** 2026-07-15
**Status:** Design complete — implementation phased (see §9)

## Motivation

Today a `KBNode` carries two orthogonal state axes plus a review gate:

| Axis | Field | Values | Question it answers |
|------|-------|--------|---------------------|
| Review gate | `status: NodeStatus` | APPROVED / PENDING_REVIEW / REJECTED | Did a human accept this node? |
| Lineage | `state: NodeState` | CURRENT / AMENDED / SUPERSEDED | Is this the latest version of this concept? |

`status` is a **pre-persistence gate**: `NodeRepository._write_nodes_and_relationships` skips any node whose `status != APPROVED` (`node_repository.py:153`), so inside the KB every node is uniformly `APPROVED`. It is not a live query axis.

Neither existing axis captures **lifecycle completion** — whether a concept has reached its terminal outcome. An open question that a later decision answers is still `CURRENT` (it is the latest version of itself) and `APPROVED`, yet it is conceptually *resolved*. The same gap exists for risks that get mitigated and (once the vocabulary supports it) decisions that get acted upon. This information currently lives only implicitly in cross-type relationship edges, forcing every consumer to traverse the graph to reconstruct it.

This spec adds a third axis — **`lifecycle`** — split into two complementary parts:

- a **persisted progress state** (`OPEN` → `RESOLVED` / `MITIGATED` / `ENACTED`), materialised on the node from the single inbound edge that completes it, queryable and RAG-filterable; and
- a **read-time overlay** (`CONFLICTED`, `INVALIDATED`) computed from live graph structure, never persisted.

The split is deliberate and is the central design decision — see §1.1 for why the two overlay states are *not* persisted.

**Phases 1–2 introduce no new relationship types.** They ride on edges the resolution agents already emit (`RESOLVES`, `MITIGATES`, `SUPERSEDES`, `CONFLICTS_WITH`). The one progress state with no triggering edge in the current schema — `ENACTED` — is deferred to Phase 3, which adds a single new relationship type (`IMPLEMENTS`). See §0.

## 0. Verified Relationship Inventory

Every lifecycle trigger must key off an edge the pipeline actually produces. The complete set of `(source_type → target_type)` pairs the resolution agents emit today, verified against the same-type `Literal` type sets (`same_type/*.py`) and all cross-type agent entry types + prompts (`cross_type/*.py`):

| Rel type | Produced pairs (source → target) |
|----------|----------------------------------|
| `SUPERSEDES` | Decision→Decision, ActionItem→ActionItem |
| `AMENDS` | Decision→Decision, ActionItem→ActionItem, Risk→Risk, OpenQuestion→OpenQuestion |
| `CONFLICTS_WITH` | Decision→Decision, ActionItem→ActionItem |
| `DEPENDS_ON` | ActionItem→ActionItem, OpenQuestion→OpenQuestion *(same-type only)* |
| `MITIGATES` | Decision→Risk, ActionItem→Risk |
| `RESOLVES` | Decision→OpenQuestion |
| `BLOCKS` | Risk→{Decision, OpenQuestion, ActionItem}, Decision→ActionItem, OpenQuestion→{Decision, ActionItem}, ActionItem→ActionItem |

**Consequences that shape this design:**

- **`RESOLVED`** ✓ — OpenQuestion is the target of `Decision→OpenQuestion RESOLVES`.
- **`MITIGATED`** ✓ — Risk is the target of `Decision→Risk` / `ActionItem→Risk MITIGATES`.
- **`CONFLICTED`** ✓ — but **only Decision and ActionItem** can ever be conflicted; Risk and OpenQuestion same-type agents emit no `CONFLICTS_WITH` (`risk.py` emits only `AMENDS`; `open_question.py` emits `AMENDS` + `DEPENDS_ON`). Permanent scope boundary, not an oversight.
- **`ENACTED`** ✗ — requires an ActionItem→Decision edge. **No such edge exists** (verified: the cross-type registry has no `(ACTION_ITEM, DECISION)` pair; the only Decision↔ActionItem edge is `Decision→ActionItem BLOCKS` — wrong direction, wrong meaning). `ENACTED` cannot fire until Phase 3 adds an `IMPLEMENTS` (ActionItem→Decision) edge.

---

## 1. Data Model

### 1.1 The two-part split — and why overlay states are not persisted

The six lifecycle values fall into two groups with fundamentally different mechanics:

| Group | Values | Driven by | Persisted? |
|-------|--------|-----------|:----------:|
| **Progress** | `OPEN`, `RESOLVED`, `MITIGATED`, `ENACTED` | one **inbound** edge where the node is the **target** | ✅ column |
| **Overlay** | `CONFLICTED`, `INVALIDATED` | the node's **surrounding live graph** (partner/ancestor state) | ❌ read-time |

**Why progress states persist cleanly.** Each is triggered by a single inbound edge (`RESOLVES`→question, `MITIGATES`→risk, `IMPLEMENTS`→decision), they are mutually exclusive by concept type, and the affected node is always the edge's `target_id`. This maps **directly** onto the existing target-keyed `_STATE_TRANSITIONS` mechanism (`node_repository.py:17`) — a parallel `rel_type → LifecycleState` map with the same idempotent write and the same `count_inbound_relationships` revert (`:229`). No recompute-from-all-edges, no both-endpoints logic.

**Why `CONFLICTED` must NOT persist.** `CONFLICTS_WITH` is one *directed* edge, but a conflict makes **both** endpoints conflicted, and a conflict is only "live" while **both** parties are `state=CURRENT` (2026-04-21 spec §4). Persisting it would require: (a) a both-endpoints write path (the target-keyed map can't express it); (b) fetching each partner's lineage `state` to judge liveness; and (c) — the trap — when a later run *supersedes* one partner, the `SUPERSEDES` edge only touches that partner, so the *other* node's persisted `CONFLICTED` would go **stale forever** unless the supersession path also revisited its conflict partners. The codebase already sidesteps all of this: it computes conflict-liveness **at read time** (Screen 4, `GET /graph/{id}` filter out `CONFLICTS_WITH` where either party `state != CURRENT`). We reuse that, so `CONFLICTED` is correct-by-construction and reverts for free.

**Why `INVALIDATED` must NOT persist.** It is inherently transitive (a dependent of a superseded ancestor). Materialising it at write time requires walking the dependency graph *inside* the write transaction — but the existing graph reads (`get_neighbours`, `count_inbound_relationships`, `query`) read from a separate pooled connection and **cannot see edges written but uncommitted in the same transaction** (`pg_store.py:375/223/413` — none accept `conn`). Computing it at read time via the existing cycle-guarded `traverse_impact` BFS (`graph.py:114`) avoids new `conn`-aware traversal machinery entirely.

**Net effect:** the persisted `lifecycle` column is a simple, idempotent, single-edge progress axis; conflict and invalidation are derived. The write path stays as simple as the lineage transitions it rides beside.

### 1.2 `LifecycleState` enum

New enum in `core/models/enums.py`. The enum holds all six values (the type is shared by the persisted column and the derived overlay), plus classmethods that partition and combine them:

```python
from collections.abc import Iterable

class LifecycleState(StrEnum):
    # -- Persisted progress axis (one inbound edge; mutually exclusive by type)
    OPEN = auto()          # tracked; no completing edge yet — initial value for every node
    RESOLVED = auto()      # OpenQuestion: target of a RESOLVES edge          (Phase 1)
    MITIGATED = auto()     # Risk: target of a MITIGATES edge                 (Phase 1)
    ENACTED = auto()       # Decision: target of an IMPLEMENTS edge           (Phase 3)
    # -- Read-time overlay (derived from live graph; never written to the column)
    CONFLICTED = auto()    # in a live CONFLICTS_WITH pair                    (Phase 1, derived)
    INVALIDATED = auto()   # dependent of a SUPERSEDED node                   (Phase 2, derived)

    @classmethod
    def progress_states(cls) -> frozenset["LifecycleState"]:
        """The persisted axis — the only values the `lifecycle` column ever holds."""
        return frozenset({cls.OPEN, cls.RESOLVED, cls.MITIGATED, cls.ENACTED})

    @classmethod
    def _precedence(cls) -> tuple["LifecycleState", ...]:
        """Lowest → highest severity, used to collapse the persisted progress state
        and the read-time overlay into a single *effective* lifecycle for display."""
        return (cls.OPEN, cls.RESOLVED, cls.MITIGATED, cls.ENACTED, cls.CONFLICTED, cls.INVALIDATED)

    @classmethod
    def dominant(cls, states: Iterable["LifecycleState"]) -> "LifecycleState":
        """Highest-precedence state among those supplied. Empty → OPEN.
        Used at READ time to merge the persisted progress state with the overlay."""
        order = cls._precedence()
        return max(states, key=order.index, default=cls.OPEN)

    @classmethod
    def rag_retrievable(cls) -> frozenset["LifecycleState"]:
        """Progress states whose nodes remain candidates for RAG seeding.
        Filters the persisted column only; overlay states never gate RAG (see §3)."""
        return frozenset({cls.OPEN, cls.ENACTED})

    @classmethod
    def rag_excluded(cls) -> frozenset["LifecycleState"]:
        return cls.progress_states() - cls.rag_retrievable()   # {RESOLVED, MITIGATED}

    @property
    def is_rag_retrievable(self) -> bool:
        return self in self.__class__.rag_retrievable()
```

**Precedence rationale** (`OPEN < RESOLVED < MITIGATED < ENACTED < CONFLICTED < INVALIDATED`):

- The progress states are mutually exclusive by concept type, so their relative order is inert — documented only for totality.
- Overlay states outrank progress: an `ENACTED` decision that is *also* contested shows as `CONFLICTED` at read time (a live dispute matters more to a reader than settled progress). `INVALIDATED` outranks `CONFLICTED` — a dead premise trumps a live dispute.
- `dominant()` runs at **read time** to produce the *effective* lifecycle: `dominant({persisted_progress} ∪ {overlay states that currently apply})`. The persisted column never holds an overlay value.

### 1.3 `KBNode`

`KBNode` in `core/models/nodes.py` gains one **persisted** field, holding only a progress state:

```python
lifecycle: LifecycleState = LifecycleState.OPEN   # persisted; always a progress state
```

Placed next to `state`. Every node starts `OPEN`. Action items carry a lifecycle too, but only `OPEN` on the persisted axis (no completing edge targets an action item until Phase 3's `IMPLEMENTS` targets decisions, not action items) — their `CONFLICTED`/`INVALIDATED` are read-time overlays like everyone else's.

The **effective** lifecycle (progress ∪ overlay) is exposed as a computed field on the API response model, not stored:

```python
# on the read/response model, not persisted
def effective_lifecycle(node, overlay: set[LifecycleState]) -> LifecycleState:
    return LifecycleState.dominant({node.lifecycle, *overlay})
```

### 1.4 `NodeFilter`

`NodeFilter` in `core/models/api_graph.py` gains:

```python
lifecycle_in: list[LifecycleState] | None = Field(
    default=None,
    description="Include only nodes whose PERSISTED lifecycle is one of these values. "
                "Filters the progress column only — CONFLICTED/INVALIDATED are read-time "
                "overlays and are not filterable here. None = no lifecycle filter.",
)
```

A list because RAG retrievability is a *set* policy (owned by `rag_retrievable()`), and a list subsumes the UI single-select case. Only progress values are meaningful; passing an overlay value matches nothing (documented, not enforced). The filter compiles to set membership — see §5 for the KB (`= ANY`) vs semantic (`{"$in": ...}`) mechanics.

### 1.5 Migration

Adds a **new column** (unlike archive, which reused `status`):

```sql
ALTER TABLE knowledge_base.kb_nodes
  ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'open';
```

Pattern verified against `002_relationship_surrogate_pk.py:20` (add-column with `server_default`) and the `state` column + index in `001`. Add a matching `ix_kb_nodes_lifecycle` index for filter performance. Existing rows backfill to `'open'`; a one-shot recompute for already-related nodes is optional (§11). Add `lifecycle` to the `write_node` INSERT column list (`pg_store.py:122`) so it is not solely reliant on the server default.

---

## 2. Derivation

Two functions, matching the split. Neither needs `dominant()` in the write path.

### 2.1 Persisted progress — write-time, target-keyed, idempotent

A parallel map beside `_STATE_TRANSITIONS`:

```python
_LIFECYCLE_TRANSITIONS = {
    RelationshipType.RESOLVES:   LifecycleState.RESOLVED,    # target: OpenQuestion
    RelationshipType.MITIGATES:  LifecycleState.MITIGATED,   # target: Risk
    RelationshipType.IMPLEMENTS: LifecycleState.ENACTED,     # target: Decision (Phase 3)
}
```

On edge create, set `target.lifecycle = _LIFECYCLE_TRANSITIONS[rel_type]` (idempotent, exactly like lineage). On edge delete, if `count_inbound_relationships(target, [rel_type]) == 0`, revert to `OPEN`. This is a mechanical clone of the lineage transition path — no both-endpoints logic, no partner fetches, no read-modify-write over the full edge set.

### 2.2 Read-time overlay — computed, never persisted

```python
async def lifecycle_overlay(node_id, node_lineage_state, graph) -> set[LifecycleState]:
    overlay = set()
    # CONFLICTED: a live CONFLICTS_WITH edge (both parties CURRENT). Reuses the existing
    # stale-conflict liveness check (2026-04-21 §4) — check inbound AND outbound edges,
    # since CONFLICTS_WITH is stored as a single directed edge.
    if await graph.has_live_conflict(node_id):                       # Phase 1
        overlay.add(LifecycleState.CONFLICTED)
    # INVALIDATED: reachable from a SUPERSEDED ancestor via dependency edges.
    # Computed with the existing cycle-guarded traverse_impact BFS (graph.py:114). Phase 2.
    if await graph.has_superseded_ancestor(node_id):                 # Phase 2
        overlay.add(LifecycleState.INVALIDATED)
    return overlay
```

Reversibility is automatic: the overlay is recomputed on every read, so a cleared conflict or a re-based dependency simply stops appearing. Nothing to un-write.

---

## 3. RAG Integration

RAG seeding filters the **persisted progress column** only. Overlay states never gate RAG — by construction they can't (they aren't in the column), and that is the *correct* behaviour:

| Persisted lifecycle | RAG-retrievable? | Reasoning |
|---------------------|:----------------:|-----------|
| `OPEN` | ✅ | Active, unsettled — prime context. |
| `ENACTED` | ✅ | Live decision being acted on — not stale. *(Phase 3.)* |
| `RESOLVED` | ❌ | Answered question — historically settled. |
| `MITIGATED` | ❌ | Addressed risk — settled. |

Conflicted and invalidated nodes: a `CONFLICTED` node keeps its underlying progress state in the column (usually `OPEN`, or `ENACTED`), so it stays retrievable — correct, since a contested-but-active decision is exactly what a new extraction should see. An `INVALIDATED` node likewise retains its progress value; if we later decide dead-premise nodes should drop out of RAG, that is a read-time enrichment, explicitly deferred (§11) rather than silently wrong.

Hard exclusion from RAG seeding = union of:
- lineage `state ∈ {SUPERSEDED, AMENDED}` (already enforced — `node_retriever.py:51`), and
- persisted `lifecycle ∈ LifecycleState.rag_excluded()` = `{RESOLVED, MITIGATED}`.

```python
filter_kwargs: dict = {
    "node_type": node.type,
    "state": NodeState.CURRENT,
    "lifecycle_in": list(LifecycleState.rag_retrievable()),   # {OPEN, ENACTED}
}
```

The UI passes no `lifecycle_in`, so all nodes remain browsable/searchable.

**Known leak — neighbour expansion.** `_expand_with_neighbours` (`node_retriever.py:115`) pulls neighbours via `get_neighbours` (`pg_store.py:375`), which applies **no** `state`/`lifecycle` predicate — so a settled node adjacent to a hit can still enter context. This is a **pre-existing hole for `state`** that `lifecycle` inherits; closing it (a filter on `get_neighbours`, benefiting both axes) is a follow-up (§11), not a Phase 1 blocker.

---

## 4. Write-Path Hook Points (progress axis)

Only the persisted progress transition touches the write path; the overlay is read-time. Progress recompute hooks exactly where lineage transitions already fire:

| Edge created/deleted | Phase | Effect on persisted `lifecycle` |
|----------------------|:-----:|---------------------------------|
| `RESOLVES` (→OpenQuestion) | 1 | target → `RESOLVED`; on delete, revert to `OPEN` if no `RESOLVES` remain |
| `MITIGATES` (→Risk) | 1 | target → `MITIGATED`; symmetric revert |
| `IMPLEMENTS` (→Decision) | 3 | target → `ENACTED`; symmetric revert |

`SUPERSEDES` and `CONFLICTS_WITH` do **not** touch the persisted lifecycle column — their lifecycle meaning (`INVALIDATED`, `CONFLICTED`) is entirely read-time overlay. `SUPERSEDES` continues to drive the lineage `state` transition as it does today.

**Principle: every relationship-creating path applies both transitions.** Whenever a relationship is written — by the pipeline, by a single manual add, or by auto-resolve — it must apply the lineage `state` transition *and* the lifecycle progress transition. There is one shared transition-aware write; all creation paths route through it.

Verified write paths to hook (all already run inside a `conn` transaction):

- **`_apply_state_transitions` / `create_relationship_manual`** (`node_repository.py:123`, `:213`) — pipeline batch + single manual relationship (`POST /graph/relationships`). `create_relationship_manual` already applies lineage `_STATE_TRANSITIONS`; add the progress transition beside it.
- **`GraphService.resolve`** (`graph.py:247-252`) — **fix required.** The `POST /graph/nodes/resolve` auto-resolve endpoint currently loops the KB-only `write_relationship` (`node_repository.py:91` — no `conn`, no transition logic), so it applies *neither* lineage nor lifecycle today. Reroute its write loop through the transition-aware path (e.g. `create_relationship_manual` per relationship, or a batch equivalent) so auto-resolve gets the same lineage + lifecycle side-effects as every other creation path. This also closes a pre-existing lineage-transition gap.
- **`delete_relationship`** (`node_repository.py:229`) — extend the existing count-and-revert to the progress axis for `RESOLVES`/`MITIGATES`/`IMPLEMENTS`.
- **`delete_node`** (`node_repository.py:66`) — same revert extension.

### 4.1 Conflict is routed through supersession, not recency

A `CONFLICTS_WITH` edge means the agent judged **both nodes still active** (`decision.py:25`). It does not auto-invalidate downstream on recency — that would override the agent's own call. When a later node genuinely renders an earlier one defunct, the agent emits **`SUPERSEDES`**, and *that* (via the read-time ancestor check) yields `INVALIDATED` for dependents. Conflict only surfaces `CONFLICTED` (read-time) on the live pair for human attention; both nodes stay RAG-retrievable.

### 4.2 Cascade reach (Phase 2, read-time)

The `INVALIDATED` overlay walks dependency edges from a `SUPERSEDED` ancestor. Per §0, `DEPENDS_ON` is **same-type only**, so in Phase 2 the reachable cascade is **action-item→action-item** chains. The headline *decision→action-item* invalidation needs the `IMPLEMENTS` dependency edge and therefore activates in **Phase 3**. Because the overlay is read-time (`traverse_impact`), extending it to `IMPLEMENTS` in Phase 3 is a traversal `rel_types` change, not new write machinery.

---

## 5. Vector Store Metadata Sync (progress axis only)

Only the persisted progress state syncs to VS metadata (the overlay is never stored, so it never syncs). Mirrors the `state` pattern (see `2026-07-12-vector-store-state-sync.md`, **already implemented** — `_transition_node_state` at `node_repository.py:97`, `update_metadata` at `pgvector_store.py:269`).

1. **`_get_vector_store_metadata`** (`node_repository.py:271`) — add `"lifecycle": node.lifecycle` to the upserted dict (matching the existing `{"state": node.state}` enum convention, not `.value`). New nodes write `lifecycle = open`.
2. **New `_transition_node_lifecycle` sibling.** `_transition_node_state` is private and state-column-specific (`_kb.update_node_state` → `SET state=$1`, `pg_store.py:167`); it can't be generalised. Add `_transition_node_lifecycle(node_id, lifecycle, *, conn)` + a KB primitive `update_node_lifecycle` (→ `SET lifecycle=$1`), writing KB column and VS metadata key in one transaction.
3. **`PGVectorStore` filter support** — add `"lifecycle"` to `get_supported_filter_fields` (**mandatory**: `_build_semantic_filter` silently drops unlisted fields — `pgvector_store.py:242`). Semantic path builds a LangChain dict → `result["lifecycle"] = {"$in": [s.value for s in lifecycle_in]}` (**not** `= ANY`); sparse path (`_apply_sparse_filter`) emits SQL → `.in_(...)`. **Verify** the installed `langchain-postgres` supports `$in` — it is the one operator not currently exercised.

**VS metadata backfill.** A `$in`/`ANY` predicate excludes NULLs, so embeddings lacking a `lifecycle` key vanish from filtered RAG queries. The KB-column backfill (§1.5) is optional, but the **VS metadata backfill is not** — backfill the key or treat missing-key as retrievable (§11).

---

## 6. Service & API Surface

- **`GET /graph`**, **`GET /graph/search`** — optional repeatable `lifecycle` query param → `lifecycle_in` (progress column filter).
- **`GET /graph/{node_id}/detail`**, node list — return both the persisted `lifecycle` **and** the computed `effective_lifecycle` (progress ∪ overlay). The overlay is computed from the same neighbour/relationship data these endpoints already fetch (`get_node_relationships`, `_get_active_neighbours` — `graph.py:108-112`), so no extra round-trips for the detail view.
- **UI (Screen 4 / browser):** filter facet on the progress axis; badges show effective lifecycle. `CONFLICTED` reinforces the existing live-conflict highlight.

No new endpoints. No role changes.

---

## 7. Metrics

The persisted progress column aggregates directly (no traversal): % of risks `MITIGATED`, count of `OPEN` questions, and (later phases) `ENACTED` decisions. Overlay-based metrics (`CONFLICTED`/`INVALIDATED` counts) require the read-time computation and are better served by a dedicated query than by column aggregation — noted, not required for MVP.

---

## 8. Interaction with Archive / Delete

Progress-axis revert hooks into the existing lineage-revert points (`delete_relationship` `:229`, `delete_node` `:66`). The read-time overlay needs **no** archive/delete hooks — it recomputes from current live structure, and archived/superseded nodes naturally drop out of the liveness/ancestor checks.

**Archive prerequisite note:** `2026-07-07-node-archive.md` describes `archive_node` and `exclude_source_status` — **neither exists in the codebase yet** (only specced). This spec does not depend on archive for Phases 1–2.

---

## 9. Phasing

The `LifecycleState` enum ships **complete** in Phase 1 (all six values, `dominant`, `progress_states`, `rag_retrievable`); later phases only start *producing* values that need new machinery.

**Phase 1 — persisted progress + conflict overlay. (Existing edges only.)**
- `LifecycleState` enum + classmethods.
- `KBNode.lifecycle` (persisted progress) + `effective_lifecycle` computed on the response model.
- Alembic column + index; `NodeFilter.lifecycle_in`; KB/VS filter support.
- Write-time progress transitions: `RESOLVES`→`RESOLVED`, `MITIGATES`→`MITIGATED` (idempotent, target-keyed, with revert).
- Read-time `CONFLICTED` overlay reusing the existing live-conflict check.
- RAG retrievable-set filter (`{OPEN, ENACTED}`) in `NodeRetriever`.
- VS metadata sync for the progress column.

Ships the full read/query/RAG value for `OPEN`/`RESOLVED`/`MITIGATED` + the conflict overlay. `ENACTED` defined but not produced.

**Phase 2 — `INVALIDATED` read-time overlay.**
- `has_superseded_ancestor` via the existing `traverse_impact` BFS (cycle-guarded), walking `DEPENDS_ON` (same-type reach only — §4.2).
- Surfaced in `effective_lifecycle` / detail endpoints. No write-path or column change.

Deferred until Phase 1 lands and orphaned dependents prove to matter.

**Phase 3 — new relationship vocabulary (agent + eval rework). (The only phase touching agents/eval.)**
- **`IMPLEMENTS` (ActionItem→Decision)** — new cross-type edge + agent + ≥2 eval-corpus examples. Unlocks the `ENACTED` **write-time** progress transition *and* extends the Phase 2 `INVALIDATED` overlay to reach decisions→their implementing action items (a `rel_types` addition to the traversal).
- **`RAISES` / `SURFACES` (Decision→Risk, Decision→OpenQuestion)** — provenance edge; **non-lifecycle** (no transition, no overlay). Bundled only because it shares the agent+eval rework.

Broadening `SUPERSEDES`/`CONFLICTS_WITH` to Risk/OpenQuestion is explicitly **not** in Phase 3 (§11).

**Concurrency note.** The persisted progress transition is idempotent (`SET lifecycle=$1`), matching lineage's safety argument under raised `max_concurrent_jobs` (2026-04-21 §4). Because we did **not** adopt a read-modify-write recompute for the persisted axis, there is no lost-update concern beyond what lineage already has. (Default `api.max_concurrent_jobs=1` — `settings.py:387`.)

---

## 10. Testing

**Unit — `test_enums.py`:**
- `progress_states()` = `{OPEN, RESOLVED, MITIGATED, ENACTED}`; `rag_retrievable()` = `{OPEN, ENACTED}`; `rag_excluded()` = `{RESOLVED, MITIGATED}`.
- `dominant()`: empty→`OPEN`; `{OPEN, CONFLICTED}`→`CONFLICTED`; `{ENACTED, CONFLICTED}`→`CONFLICTED`; `{RESOLVED, INVALIDATED}`→`INVALIDATED`.

**Unit — `test_node_repository.py`:**
- P1: `RESOLVES` → target question `lifecycle=RESOLVED`; VS metadata synced.
- P1: `MITIGATES` → target risk `MITIGATED`.
- P1: delete last `RESOLVES` → revert to `OPEN`; other `RESOLVES` remaining → no revert.
- P3: `IMPLEMENTS` → target decision `ENACTED`.
- Confirm `SUPERSEDES`/`CONFLICTS_WITH` do **not** modify the persisted `lifecycle` column.

**Unit — overlay + resolve (service-level, `test_graph_service.py`):**
- P1: `has_live_conflict` true iff a `CONFLICTS_WITH` edge exists with both parties `CURRENT`; false once a party is `SUPERSEDED` (stale — reverts for free).
- P2: `has_superseded_ancestor` walks `DEPENDS_ON`, cycle-guarded, terminates.
- `effective_lifecycle` = `dominant(persisted ∪ overlay)`.
- P1: `GraphService.resolve` (auto-resolve) applies lineage **and** lifecycle transitions — a produced `RESOLVES`/`MITIGATES` sets the target's `lifecycle`, and a `SUPERSEDES` advances the target's lineage `state` (regression guard for the fixed KB-only-write gap).

**Unit — `test_pg_store.py` / VS tests:** `lifecycle_in` membership in KB `query` (`= ANY`) and both VS builders (`$in` / `.in_()`); `get_supported_filter_fields` includes `lifecycle`; `update_metadata` writes the key.

**Unit — `test_node_retriever.py`:** `retrieve` sets `lifecycle_in = {OPEN, ENACTED}`; `RESOLVED`/`MITIGATED` nodes excluded from candidates.

**Integration — `test_node_repository_state_sync.py`:** approve question (`open`) → add `RESOLVES` → assert KB + VS `lifecycle == "resolved"` → delete edge → assert reverted to `"open"`.

---

## 11. Deferred / Open

- **Neighbour-expansion filter leak.** `get_neighbours` applies no state/lifecycle predicate, so settled neighbours leak into RAG context (§3). Pre-existing for `state`; fix once for both axes.
- **VS metadata backfill.** Existing embeddings need the `lifecycle` key or they drop from membership-filtered RAG (§5).
- **Excluding `INVALIDATED` from RAG is not free (deferred).** RAG filtering is a flat predicate over **VS metadata**; because `INVALIDATED` is a read-time overlay (never in the column/metadata), the search filter is **structurally blind** to it. The only way to exclude invalidated candidates is to **post-filter** after the vector search — run `has_superseded_ancestor` on each of the (≤ `top_k × 2`) KB-fetched hits (`node_retriever.py:104`) and drop the invalidated ones — which adds N graph traversals to the retrieval hot path. **Decision: accept the leak for now.** The Phase 2 invalidated set is narrow (same-type action-item chains only until Phase 3), the "invalidated ⇒ irrelevant" inference is debatable (a dependent whose prerequisite was superseded may still be useful context), and the post-filter is a localized retriever change if evidence later shows it matters — no model or write-path impact.
- **`SUPERSEDES`/`CONFLICTS_WITH` for Risk & OpenQuestion.** Their same-type agents emit neither today, so risks/questions can't be `CONFLICTED` or drive `INVALIDATED`. Explicitly excluded from Phase 3; revisit as a separate vocabulary effort.
- **KB-column backfill** for existing edges (optional; new runs converge).

---

## 12. Relevant Files

**Phases 1–2:**
- `src/seshat/core/models/enums.py` — `LifecycleState` + classmethods.
- `src/seshat/core/models/nodes.py` — persisted `KBNode.lifecycle`.
- `src/seshat/core/models/api_graph.py` — `NodeFilter.lifecycle_in`; `effective_lifecycle` on the response model.
- `src/seshat/app/repositories/node_repository.py` — `_LIFECYCLE_TRANSITIONS`; `_transition_node_lifecycle`; progress revert hooks; VS sync.
- `src/seshat/app/services/graph.py` — `has_live_conflict` (P1), `has_superseded_ancestor` via `traverse_impact` (P2), `effective_lifecycle` in detail/list responses.
- `src/seshat/infra/knowledge_store/pg_store.py` — `lifecycle` column + index; `update_node_lifecycle`; `lifecycle_in` `= ANY` predicate.
- `src/seshat/infra/vector_store/pgvector_store.py` — `lifecycle` in `get_supported_filter_fields`; `$in` (semantic) + `.in_()` (sparse).
- `src/seshat/app/pipeline/extraction/node_retriever.py` — `lifecycle_in = rag_retrievable()`.
- `alembic/` — `lifecycle` column migration.
- `tests/` — as in §10.

**Phase 3 only (new vocabulary — untouched in Phases 1–2):**
- `src/seshat/core/models/enums.py` — `RelationshipType.IMPLEMENTS`, `RAISES`/`SURFACES`.
- `src/seshat/app/agents/resolution/cross_type/` — new agent(s) for ActionItem→Decision (`IMPLEMENTS`) and Decision→Risk/OpenQuestion (`RAISES`); register in the cross-type registry.
- `data/eval/` — ≥2 labelled corpus examples per new rel type.
- `docs/superpowers/specs/2026-04-21-seshat-design.md` §4 — add the new pairings to the cross-type schema table.
