from pathlib import Path
from typing import ClassVar

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from seshat.core.config.settings import DEFAULT_EVAL_GATE_PATH, PROJECT_ROOT
from seshat.core.models.enums import EvalHarness, SearchMode

_DEFAULT_CORPUS_BASE_DIR: Path = PROJECT_ROOT / "data" / "eval" / "corpora"


class EvalConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EVAL__",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    corpus_base_dir: Path = Field(
        default=_DEFAULT_CORPUS_BASE_DIR,
        description="Root directory for eval corpora. Expected subdirs: one per eval harness.",
    )
    gate_path: Path = Field(
        default=DEFAULT_EVAL_GATE_PATH,
        description="Full path (including filename) for the GateResult JSON output.",
    )
    run_identification: bool = Field(
        default=True,
        description=(
            "Run the identification eval pass, i.e., "
            "check if the pipeline extracted the right nodes from the transcript."
        ),
    )
    run_resolution: bool = Field(
        default=True,
        description=(
            "Run the resolution eval pass, i.e., "
            "check if the pipeline inferred the correct relationships between nodes."
        ),
    )
    run_retrieval: bool = Field(
        default=True,
        description=(
            "Run the retrieval eval pass, i.e., "
            "check if the fully assembled retrieval pipeline (search + rerank, as configured) "
            "surfaces the right nodes (similar and related neighbors)."
        ),
    )
    run_vector_search: bool = Field(
        default=True,
        description=(
            "Run the vector_search eval pass, i.e., "
            "check if pure dense-embedding similarity surfaces the right nodes, "
            "isolated from keyword extraction, multi-query, and reranking."
        ),
    )
    run_sparse_search: bool = Field(
        default=True,
        description=(
            "Run the sparse_search eval pass, i.e., "
            "check if pure full-text (ts_rank_cd) similarity surfaces the right nodes, "
            "isolated from keyword extraction."
        ),
    )
    run_grounding: bool = Field(
        default=True,
        description=(
            "Run the grounding eval pass, i.e., "
            "check if the grounding agent correctly identifies grounded vs. hallucinated descriptions."
        ),
    )
    run_grouping: bool = Field(
        default=True,
        description=(
            "Run the grouping eval pass, i.e., "
            "check if the grouping agent correctly clusters extracted items into thematic groups."
        ),
    )
    max_concurrent_predictions: int = Field(
        default=10,
        gt=0,
        description="Maximum number of prediction coroutines that may run in parallel during eval.",
    )
    # TODO(Tier 3): remove once the composite retrieval harness is rewritten around
    # NodeRetriever.retrieve() (no raw scores to threshold-filter) — see docs/superpowers/specs/
    # 2026-07-31-eval-harness-expansion-design.md §3. Until then, eval/retrieval/runner.py still
    # uses this field as-is.
    retrieval_score_thresholds: dict[SearchMode, float] = Field(
        default_factory=dict,
        description="Per-mode minimum score thresholds [0, 1] applied during retrieval eval.",
    )
    # Dense-leg score thresholds calibrated by the vector_search meta-scorer (argmax macro-F2).
    # Absent keys default to 0.0 (no filtering). Keyed by SearchMode for forward-compat with the
    # meta-scorer's mode-agnostic sweep logic, but vector_search always runs in SEMANTIC mode.
    # Set via EVAL__VECTOR_SEARCH_SCORE_THRESHOLDS__SEMANTIC=0.77 etc.
    vector_search_score_thresholds: dict[SearchMode, float] = Field(
        default_factory=dict,
        description="Per-mode minimum cosine-similarity thresholds [0, 1] applied during vector_search eval.",
    )
    # Sparse-leg score thresholds calibrated by the sparse_search meta-scorer (argmax macro-F2).
    # ts_rank_cd has its own score scale distinct from cosine similarity, so this is calibrated
    # independently from vector_search_score_thresholds. sparse_search always runs in KEYWORD mode.
    # Set via EVAL__SPARSE_SEARCH_SCORE_THRESHOLDS__KEYWORD=0.05 etc.
    sparse_search_score_thresholds: dict[SearchMode, float] = Field(
        default_factory=dict,
        description="Per-mode minimum ts_rank_cd thresholds [0, 1] applied during sparse_search eval.",
    )

    # a hidden folder in the project root for caching intermediate results during eval runs; not intended for manual use
    _cache_dir: ClassVar[Path] = PROJECT_ROOT / ".seshat" / "eval_cache"

    def corpus_dir(self, harness: EvalHarness) -> Path:
        """Return the corpus directory for a harness (relative to the configured corpus_base_dir)."""
        return self.corpus_base_dir / harness

    @classmethod
    def cache_dir(cls, harness: EvalHarness) -> Path:
        """Return the cache directory for a harness, without constructing an instance."""
        return cls._cache_dir / harness

    @property
    def enabled_harnesses(self) -> list[EvalHarness]:
        """Harnesses whose run_<harness> flag is enabled, in canonical EvalHarness declaration order."""
        flags = {
            EvalHarness.IDENTIFICATION: self.run_identification,
            EvalHarness.RESOLUTION: self.run_resolution,
            EvalHarness.VECTOR_SEARCH: self.run_vector_search,
            EvalHarness.SPARSE_SEARCH: self.run_sparse_search,
            EvalHarness.RETRIEVAL: self.run_retrieval,
            EvalHarness.GROUNDING: self.run_grounding,
            EvalHarness.GROUPING: self.run_grouping,
        }
        return [h for h in EvalHarness if flags[h]]

    @field_validator("gate_path", mode="after")
    @classmethod
    def _validate_gate_path(cls, v: Path) -> Path:
        if v.suffix != ".json":
            raise ValueError(f"gate_path must be a .json file, got: {v}")
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _validate_corpus_dirs(self) -> "EvalConfig":
        for harness in self.enabled_harnesses:
            path = self.corpus_dir(harness)
            if not path.is_dir():
                raise ValueError(f"corpus dir does not exist: {path}")
        return self

    @model_validator(mode="after")
    def _create_cache_dirs(self) -> "EvalConfig":
        for harness in EvalHarness:
            self.cache_dir(harness).mkdir(parents=True, exist_ok=True)
        return self
