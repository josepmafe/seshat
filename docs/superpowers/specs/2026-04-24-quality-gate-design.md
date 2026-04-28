# Quality Gate — Design Spec

**Date:** 2026-04-24
**Status:** Approved

## Overview

A two-layer quality gate for Seshat: pre-commit hooks for fast local feedback on every commit, and a GitHub Actions workflow for full CI enforcement on every push/PR. Linting, formatting, type checking, and unit tests are hard blockers. Complexity metrics and coverage are informational, reported in the GHA Job Summary on PRs.

---

## 1. Pre-commit Hooks

File: `.pre-commit-config.yaml`

Three hooks, all hard blockers:

| Hook | Command | Purpose |
|------|---------|---------|
| ruff lint | `ruff check .` | Style, unused imports, security anti-patterns (`S` rules) |
| ruff format | `ruff format --check .` | Format check — dev runs `ruff format` to fix locally |
| mypy | `mypy src/` | Static type check against `src/` only |

pytest is **not** in pre-commit — tests can be slow and shouldn't block every commit.

---

## 2. GitHub Actions Workflow

File: `.github/workflows/ci.yml`

**Triggers:**
- `push` — runs `quality` job only
- `pull_request` — runs both `quality` and `report` jobs

Runner: `ubuntu-latest` (free tier Linux, unlimited minutes on public repos; 2,000 min/month on private).

### Job 1: `quality` (hard block)

Runs on every push and every PR update.

Steps:
1. Checkout + `uv sync`
2. `pre-commit run --all-files` — confirms hooks pass in CI regardless of local setup
3. `uv run mypy src/` — type check
4. `uv run pytest tests/unit/ -m "not integration"` — unit tests; integration tests require LocalStack/Chroma and are out of scope for MVP CI

### Job 2: `report` (informational, PR only)

Runs only on `pull_request`. Never fails the workflow — outputs are written to the GHA Job Summary tab.

Steps:
1. Checkout + `uv sync`
2. `uv run radon cc src/ -s -a` — cyclomatic complexity: per-function scores + module averages
3. `uv run radon mi src/` — maintainability index: letter grade per file
4. `uv run pytest tests/unit/ -m "not integration" --cov=src/seshat --cov-report=term-missing` — coverage report
5. All outputs appended to `$GITHUB_STEP_SUMMARY`

Cognitive complexity (SonarQube-proprietary) is not included — radon does not implement it and pulling in a separate tool is not justified for an informational report.

---

## 3. Tool Configuration (`pyproject.toml`)

### mypy

```toml
[tool.mypy]
files = ["src"]
disallow_untyped_defs = true
ignore_missing_stubs = true
```

Strict mode is too noisy for an MVP. `disallow_untyped_defs` enforces typed function signatures; `ignore_missing_stubs` prevents failures on third-party libraries without stubs.

### pytest

```toml
[tool.pytest.ini_options]
markers = ["integration: requires external services (LocalStack, Chroma, MLflow)"]
```

Integration tests are marked explicitly. CI runs with `-m "not integration"` to skip them. Running integration tests locally requires the full Docker Compose stack.

### Coverage

No hard minimum threshold enforced for MVP. To add one later: `--cov-fail-under=N` in the pytest command (or `fail_under` in `[tool.coverage.report]`).

---

## 4. Scope Notes

- `report` job runs on PRs only — no report generated on pushes to `main` after merge (MVP scope).
- No PR comments — GHA Job Summary is sufficient and requires no extra permissions or wiring.
- No coverage badge or artifact uploads — out of scope for MVP.
- Integration tests in CI (with Docker Compose service containers) deferred to v2.
