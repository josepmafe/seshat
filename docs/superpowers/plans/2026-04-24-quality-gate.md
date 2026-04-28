# Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-layer quality gate: pre-commit hooks (ruff lint, ruff format, mypy) and a GHA workflow (quality hard-block job + informational report job on PRs).

**Architecture:** Pre-commit runs fast local checks on every commit. GHA re-runs pre-commit in CI and adds pytest (unit only) as a hard block; a separate `report` job runs only on PRs and writes radon complexity + pytest coverage to the GHA Job Summary. All tool config lives in `pyproject.toml`.

**Tech Stack:** `pre-commit`, `ruff`, `mypy`, `pytest`, `pytest-cov`, `radon`, GitHub Actions (`ubuntu-latest`)

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `.pre-commit-config.yaml` | Create | Defines ruff lint, ruff format, mypy hooks |
| `pyproject.toml` | Modify | Add `[tool.mypy]` and `[tool.pytest.ini_options]` sections; add dev dependencies |
| `.github/workflows/ci.yml` | Create | GHA workflow: `quality` + `report` jobs |

---

### Task 1: Add dev dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Check current pyproject.toml state**

Run: `cat pyproject.toml`

You need to know the existing structure before editing. Look for `[project.optional-dependencies]` or `[dependency-groups]` sections — uv projects typically use `[dependency-groups]`.

- [ ] **Step 2: Add dev dependencies**

Add a `[dependency-groups]` section (or extend the existing one) with the required dev tools:

```toml
[dependency-groups]
dev = [
    "mypy>=1.10",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "radon>=6.0",
    "pre-commit>=3.7",
]
```

If the section already exists, merge these entries into it — don't create a duplicate section.

- [ ] **Step 3: Add mypy configuration**

Append to `pyproject.toml`:

```toml
[tool.mypy]
files = ["src"]
disallow_untyped_defs = true
ignore_missing_stubs = true
```

- [ ] **Step 4: Add pytest configuration**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["integration: requires external services (LocalStack, Chroma, MLflow)"]
```

- [ ] **Step 5: Sync dependencies**

Run: `uv sync`

Expected: uv resolves and installs the new dev dependencies without errors.

- [ ] **Step 6: Verify tools are available**

Run: `uv run mypy --version && uv run pytest --version && uv run radon --version`

Expected: version strings printed for all three tools.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add dev dependencies for quality gate (mypy, pytest, radon, pre-commit)"
```

---

### Task 2: Configure pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        args: [src/]
        pass_filenames: false
```

> **Note:** Check the latest stable tags for `ruff-pre-commit` and `mirrors-mypy` at https://github.com/astral-sh/ruff-pre-commit/releases and https://github.com/pre-commit/mirrors-mypy/tags before committing — pin to the latest stable, not the versions above if they're outdated.

- [ ] **Step 2: Install the hooks**

Run: `uv run pre-commit install`

Expected:
```
pre-commit installed at .git/hooks/pre-commit
```

- [ ] **Step 3: Run hooks against all files to verify they pass**

Run: `uv run pre-commit run --all-files`

Expected: all hooks pass (ruff, ruff-format, mypy). If mypy fails because `src/` is empty or has no Python files yet, that's fine — the hook config is correct. If ruff fails on existing files, fix the reported issues before continuing.

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hooks (ruff lint, ruff format, mypy)"
```

---

### Task 3: Create the GHA workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the `.github/workflows/` directory**

Run: `mkdir -p .github/workflows`

- [ ] **Step 2: Create `ci.yml`**

```yaml
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["main"]

jobs:
  quality:
    name: Quality (lint, type check, tests)
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync

      - name: Run pre-commit
        run: uv run pre-commit run --all-files

      - name: Type check
        run: uv run mypy src/

      - name: Unit tests
        run: uv run pytest tests/unit/ -m "not integration" -v

  report:
    name: Report (complexity, coverage)
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync

      - name: Cyclomatic complexity
        run: |
          echo "## Cyclomatic Complexity" >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
          uv run radon cc src/ -s -a >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY

      - name: Maintainability index
        run: |
          echo "## Maintainability Index" >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
          uv run radon mi src/ >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY

      - name: Coverage
        run: |
          echo "## Test Coverage" >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
          uv run pytest tests/unit/ -m "not integration" --cov=src/seshat --cov-report=term-missing 2>&1 >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
```

> **Note on `python-version`:** use the same Python version as your local dev environment. Run `python --version` locally to confirm — replace `3.12` if different.

- [ ] **Step 3: Verify the YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "valid"`

Expected: `valid`

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GHA quality gate (quality + report jobs)"
git push
```

- [ ] **Step 5: Verify the workflow triggers**

Go to the repository's **Actions** tab on GitHub. You should see a `CI` workflow run triggered by the push. Confirm the `quality` job completes (green or expected failure if `tests/unit/` doesn't exist yet — see Task 4).

---

### Task 4: Add a smoke test so the `quality` job doesn't fail on missing test directory

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_smoke.py`

The `quality` job runs `pytest tests/unit/`. If that directory doesn't exist, pytest exits with an error and blocks CI. Add a minimal smoke test to keep CI green until real tests are written.

- [ ] **Step 1: Create the test directory structure**

Run: `mkdir -p tests/unit`

- [ ] **Step 2: Create `__init__.py` files**

Run: `touch tests/__init__.py tests/unit/__init__.py`

- [ ] **Step 3: Create `tests/unit/test_smoke.py`**

```python
def test_smoke():
    assert True
```

- [ ] **Step 4: Run the smoke test locally**

Run: `uv run pytest tests/unit/test_smoke.py -v`

Expected:
```
tests/unit/test_smoke.py::test_smoke PASSED
```

- [ ] **Step 5: Commit and push**

```bash
git add tests/
git commit -m "test: add smoke test to keep CI green until real tests land"
git push
```

- [ ] **Step 6: Verify CI passes**

Check the **Actions** tab on GitHub. The `quality` job should now be fully green.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Covered by |
|-----------------|-----------|
| Pre-commit: ruff lint | Task 2 |
| Pre-commit: ruff format | Task 2 |
| Pre-commit: mypy | Task 2 |
| GHA `quality` job: pre-commit, mypy, pytest unit | Task 3 |
| GHA `report` job: radon cc + mi, coverage | Task 3 |
| `report` only on PR | Task 3 (`if: github.event_name == 'pull_request'`) |
| GHA Job Summary output | Task 3 |
| `[tool.mypy]` config | Task 1 |
| `[tool.pytest.ini_options]` markers | Task 1 |
| pytest not in pre-commit | Task 2 (absent by design) |

All spec requirements covered. No gaps.

**Placeholder scan:** None found.

**Type consistency:** No shared types across tasks — each task is self-contained config/infra.
