# Seshat — NLI Faithfulness as a Second Confidence Signal

**Date:** 2026-06-10
**Status:** Draft — pending orthogonality verification with real NLI scores. Refreshed 2026-07-20: aligned with the shipped code (the confidence gate is the *grounding* agent, not a "verification" one; `IdentificationMetaScorer` has no weight-calibration loop yet — see §3.4).

---

## 1. Motivation

The pipeline currently computes node confidence from two signals:

- **Heuristics** (structural): quote anchor presence, required field population, field completeness. A weighted composite of three sub-signals whose weights (`_W_QUOTE=0.3`, `_W_TITLE=0.3`, `_W_DESC=0.4`, `heuristics_scorer.py:33`) are hardcoded constants — commented in-code as "hand-tuned starting points with no empirical basis." They are **not** currently calibrated against the corpus (see §3.4).
- **Grounding** (semantic, binary gate): an optional second LLM agent (`GroundingLLMConfig`, `settings.py:97`; the `grounding` block, `settings.py:153`) that approves or rejects a node. Expensive — one API call per node. Disabled by default (`grounding=None`), and a validator forces its provider to differ from identification's (`settings.py:195`).

A preliminary orthogonality analysis (`development/nli_analysis/`) using `token_set_ratio` as a
faithfulness proxy showed r ≈ 0 between heuristic score and lexical faithfulness across 75 cached
nodes. This suggests heuristics and faithfulness are independent dimensions: a structurally complete
node can still produce a description or title that misrepresents the cited quote.

**The gap:** heuristics cannot catch a node that anchors correctly, populates all fields, but writes
a description that contradicts or hallucinates content not present in the quoted span. The
grounding agent catches this, but at LLM-call cost for every node — and only when grounding is
enabled at all. A cheap, continuous faithfulness signal could reduce grounding agent calls while
adding an independent quality dimension to the confidence score.

---

## 2. Proposed Signal: NLI Entailment

**Model:** `cross-encoder/nli-deberta-v3-small` (~180MB, ~50ms/node, local inference, no API calls).

For each extracted node, compute:
- `nli_description`: entailment probability of `(premise=quote_span, hypothesis=description)`
- `nli_title`: entailment probability of `(premise=quote_span, hypothesis=title)`

Both scores are in [0, 1]. High entailment = description/title is supported by the quoted evidence.
Low entailment or high contradiction = potential hallucination.

**Why NLI over token_set_ratio:** `token_set_ratio` penalizes legitimate paraphrases equally to
hallucinations. NLI distinguishes entailment from contradiction — a description that uses different
words but is semantically consistent with the quote scores high, while one that contradicts the
quote scores low.

---

## 3. Architecture

### 3.1 Current confidence model

```
confidence = heuristics                          # sole continuous signal
approved = grounding_passed AND heuristics >= threshold
```

`KBNode.confidence` is set directly to the heuristics score (`confidence=self.breakdown.heuristics`,
`pending_node.py:57`). Grounding is a **hard binary gate** (`assign_status`, `pending_node.py:92-117`):
`grounding_passed is None` (grounding disabled or retries exhausted) counts as passing, so with
grounding off the gate reduces to `heuristics >= threshold`. When grounding is on and a node fails
it, the node is rejected entirely in auto mode (or held in `PENDING_REVIEW` in manual mode),
regardless of its heuristic score. It does not blend into confidence arithmetically.
`ConfidenceBreakdown` today carries exactly `grounding_enabled`, `grounding_passed`, and `heuristics`
(`nodes.py:30-41`).

### 3.2 Proposed confidence model

NLI would be a second continuous input into the confidence score alongside heuristics:

```
confidence = w_heuristics * heuristics + w_nli * nli_score
approved = grounding_passed AND confidence >= threshold
```

Where `nli_score = mean(nli_description, nli_title)` — simple mean of the two NLI signals.
Rationale: heuristics already aggregates 3 sub-signals (quote length, title specificity, description
directness); NLI has 2 (description, title). Simple mean keeps the aggregation consistent in spirit
across both dimensions without introducing additional tunable parameters.

Grounding remains a hard gate — NLI does not replace it. Note the gate compares against `heuristics`
today (`pending_node.py:93`); moving it to the blended `confidence` above is part of this change, so
`assign_status` and the auto-approval threshold must read the composite rather than the raw heuristics
score.

`ConfidenceBreakdown` gains an optional `nli` field. When `nli_enabled=False` (default), `w_nli=0`
and the formula degrades to `confidence = heuristics`.

`w_heuristics` and `w_nli` — TBD, to be discussed before calibration.

### 3.3 Heuristics aggregation (open question)

For symmetry with the NLI simple mean, heuristics could also drop its hand-tuned sub-weights
(`_W_QUOTE=0.3`, `_W_TITLE=0.3`, `_W_DESC=0.4`) in favour of one of two approaches:

- **Simple mean:** `mean(quote_score, title_score, desc_score)` — consistent with NLI aggregation,
  eliminates three hand-tuned constants.
- **Length-weighted mean:** weight each sub-signal by the word count of its input field (quote words,
  title words, description words). A longer, richer field contributes proportionally more to the
  score. Adaptive rather than fixed — no constants to tune.

Both are worth evaluating against the labeled corpus alongside the NLI weight calibration. TBD.

### 3.4 Weight calibration

**Current state — no weight calibration exists yet.** `IdentificationMetaScorer`
(`identification_meta_scorer.py`) only sweeps a single scalar auto-approval **threshold** over `[0, 1]`
(`sweep_threshold` / `precision_coverage_curve`, lines 46-96); it never touches the heuristic
sub-weights, which are hardcoded constants (`heuristics_scorer.py:33-35`). There is no `fit_weights()`
method to extend. So this is **new capability**, not an extension of an existing loop.

**Plan:** add a `fit_weights()` method to `IdentificationMetaScorer` that optimizes the confidence-blend
weights — at minimum `w_heuristics` and `w_nli` (§3.2), and optionally the heuristic sub-weights
(`_W_QUOTE`, `_W_TITLE`, `_W_DESC`) if §3.3's simple/length-weighted-mean direction is not taken.
It reuses the same corpus-driven, file-cached prediction path `_build_cache()` already provides, so no
new evaluation infrastructure is needed — only the search over the weight space and the scoring of each
candidate.

**Deferred — the optimization objective.** Which metric `fit_weights()` maximizes (and over what search
strategy — grid sweep, coordinate ascent, or a small optimizer) is left open here, to be fixed when NLI
scoring is actually available and the orthogonality prerequisite (§5) is met. The threshold sweep uses
"argmax coverage subject to precision ≥ p_target" (`identification_meta_scorer.py:88-94`); the weight
objective may or may not reuse that criterion. This choice is deliberately not pinned down in this spec.

**Interim:** until `fit_weights()` exists and is run, ship `w_heuristics=1.0`, `w_nli=0.0` (i.e.
`nli_enabled=False`, §3.2) so behavior is identical to today. Calibrated weights are a follow-up, not a
blocker for landing the NLI signal plumbing.

### 3.5 Grounding routing

An additional use: skip the grounding agent for nodes with both high heuristic score AND high NLI
faithfulness. This reduces LLM cost on clear-cut cases without changing the gate semantics. Applies
only when grounding is enabled — with `grounding=None` (the default) there is no call to skip.

```
if heuristics >= h_threshold and nli_score >= nli_threshold:
    skip grounding  # both independent dimensions are confident
else:
    run grounding agent
```

A skipped node leaves `grounding_passed=None`, which the gate already treats as passing
(`pending_node.py:92`), so the routing needs no change to `assign_status` beyond the skip decision
itself.

**TODO:** define `nli_threshold`. TBD from calibration sweep.

**Prerequisite:** orthogonality must hold empirically with real NLI scores (not just the lexical
proxy). If r is non-trivial, the routing logic above may double-count signal.

---

## 4. Infrastructure

**Deployment context:** Seshat runs in AWS ECS. The NLI model is not a free local call:

| Option | Pros | Cons |
|---|---|---|
| Co-located in main container | Simpler ops, no network hop | +~180MB image, higher cold start and memory footprint |
| Dedicated sidecar container | Lean main container, reusable across services | Network hop per call, more ops complexity |

**TBD:** deployment option. Start with co-located for simplicity; revisit if memory is a concern.

**Current blocker:** `cross-encoder/nli-deberta-v3-small` cannot be downloaded from HuggingFace in
the corporate environment (SSL certificate verification failure). Must be resolved before
implementation. Options, best first:

1. **Point `requests` at the corporate CA** via `REQUESTS_CA_BUNDLE` (and `CURL_CA_BUNDLE`) → the
   proxy's CA `.pem`. Keeps TLS verification *on*; no insecure escape hatch. Preferred.
2. **Config-gated SSL opt-out for the HF backend.** Note the existing
   `disable_ssl_verification` patch (`core/utils/http_patch.py`) does **not** help here: it only
   monkeypatches `httpx.Client` (`http_patch.py:19`), whereas `sentence-transformers` →
   `huggingface_hub` downloads go through `requests`. The analogous fix is
   `huggingface_hub.configure_http_backend(...)` to inject a `requests.Session` with a custom CA (or,
   as a last resort on a trusted network, `verify=False`) — i.e. extend the same
   `SeshatConfig.disable_ssl_verification`-gated pattern to the `requests`/HF path rather than reuse
   the httpx patch unchanged.
3. **Pre-download the weights and bundle them in the container image** — sidesteps the network at
   runtime entirely; also removes cold-start download latency.

---

## 5. Open Questions

1. **Orthogonality with real NLI** — the lexical proxy analysis (r ≈ 0) is suggestive but not
   conclusive. Must be verified once the model is accessible. If r is high, NLI is redundant and
   this spec is moot.

2. **Paraphrase vs hallucination threshold** — NLI entailment scores vary with phrasing style. The
   agent may consistently produce high-quality paraphrases that score lower than verbatim copies.
   Calibrate `nli_threshold` on the eval corpus before using it as a gate.

3. **Effect on FP rate** — the 20 high-heuristic/low-lexical-faithfulness nodes in the current
   corpus need manual inspection. If most are legitimate paraphrases, adding NLI to confidence
   may hurt precision without reducing FP rate. Inspect before committing to implementation.

---

## 6. What This Does Not Cover

- **NLI as an offline eval scorer** — using NLI to judge agent output quality in the eval harness is a
  separate, independent use from the live pipeline confidence scoring this spec concerns. (This was
  covered in `2026-05-24-seshat-eval-quality-scoring.md`, since removed from the repo, so the
  cross-reference is dangling and the eval-scorer use is currently unspecified — noted here only to
  scope it out.)
- **HeuristicsScorer sub-weight tuning** — the `_W_QUOTE`/`_W_TITLE`/`_W_DESC` constants
  (`heuristics_scorer.py:33`) are hand-tuned and uncalibrated today. Whether to calibrate them (vs
  replace them with a parameter-free mean) is raised as an open question in §3.3 and folded into the
  `fit_weights()` plan in §3.4 — it is not a pre-existing, already-scoped workstream.
- **Resolution calibration** — not affected. (There is no `ResolutionMetaScorer` in the codebase; the
  calibration package has only `IdentificationMetaScorer` and `RetrievalMetaScorer`.)
