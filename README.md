# Seshat

Seshat is a GenAI pipeline that turns meeting recordings into a structured knowledge base. It ingests audio or
pre-formatted transcripts, extracts Architecture Decision Records, risks, agreements, and action items, and writes
them to a queryable store that tracks relationships between decisions across meetings — supersessions, amendments,
and conflicts.

Built as a master's thesis project.

## Running it

Seshat ships fully containerized: you only need `docker compose` to run it.

```bash
# 1. Copy the env templates
cp .env.example .env
cp docker/.env.docker.example .env.docker

# 2. Fill in your API keys in .env.docker (and keep .env in sync).
#    Minimum: SESHAT_ROOT_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, ASSEMBLYAI_API_KEY.

# 3. Bring up the whole stack
docker compose up
```

Once everything is healthy:

- **Web UI** → http://localhost:8501
- **REST API** → http://localhost:8000 (docs at `/docs`)
- **MLflow** → http://localhost:5000

For the full walkthrough —getting an API key, submitting jobs, reviewing extractions, and exploring the graph—
see [`docs/user-guide.md`](docs/user-guide.md).

## Local development

Working outside the containers (tests, eval harnesses, or running a service directly) needs **Python 3.13+**
and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                    # install the project and dev dependencies
uv sync --group eval       # add the optional eval dependencies
```

Local runs read config from `.env` (loaded via `python-dotenv`), with `SECRETS__PROVIDER=env` so secrets come
straight from environment variables instead of Secrets Manager. Bring up the infra containers you need
(`docker compose up postgres mlflow localstack`), then use the CLI:

```bash
uv run seshat migrate      # apply DB migrations
uv run seshat api          # serve the API (defaults to :8000)
```

Running and calibrating the evaluation harnesses has its own workflow — dependency group, MLflow, the prediction
cache, and the release gate. See [`docs/eval-user-guide.md`](docs/eval-user-guide.md).

### Running tests

The default `uv run pytest` run excludes the `llm` marker (see `addopts` in `pyproject.toml`). Use these commands
depending on what you want to cover:

| Command | What runs |
|---------|-----------|
| `uv run pytest` | Default (excludes llm) |
| `uv run pytest -m ""` | All tests |
| `uv run pytest -m "not integration"` | Pure unit tests |
| `uv run pytest -m integration` | Integration tests (includes llm) |
| `uv run pytest -m "integration and not llm"` | Non-LLM integration (Postgres, LocalStack, MLflow) |
| `uv run pytest -m llm` | Tests requiring a live LLM API key |
| `uv run pytest -m agents` | Agent tests (subset of llm) |
| `uv run pytest -m embedding` | Embedding tests (subset of llm) |

**LLM tests cost money.** Always redirect output to a file when running them:

```bash
uv run pytest -m llm 2>&1 | tee /tmp/llm_test_out.txt
```

### Common uv commands

| Command | Description |
|---------|-------------|
| `uv run pytest` | Run tests (according to the default addopts defined in `pyproject.toml`) |
| `uv run ruff check src` | Lint |
| `uv run ruff format src` | Format |
| `uv run mypy` | Type check |
| `uv run radon cc src -a -nb` | Cyclomatic complexity (shows B+ rated items only) |
| `uv sync --group eval` | Install optional eval dependencies |

## Project structure

```
seshat/
├── src/
│   ├── seshat/
│   │   ├── core/                        # Pure data and config — no I/O, no AI
│   │   │   ├── models/                  # Pydantic domain models (KBNode, enums, …)
│   │   │   ├── config/                  # Pydantic settings (SeshatConfig, LLMConfig, ExtractionConfig, …)
│   │   │   └── utils/                   # Shared pure utilities (audio, retry, tokens, logging)
│   │   ├── infra/                       # External system adapters — I/O only, no business logic
│   │   │   ├── blob_store/              # S3 blob store abstraction (aiobotocore)
│   │   │   ├── vector_store/            # pgvector semantic search abstraction
│   │   │   ├── knowledge_store/         # Postgres-backed KB node persistence
│   │   │   ├── ops_store/               # Postgres-backed job/ops ledger
│   │   │   └── secrets/                 # AWS Secrets Manager helpers
│   │   ├── app/                         # Runtime application — orchestration, AI, and services
│   │   │   ├── agents/                  # LLM agents
│   │   │   │   ├── identification/      # Extraction agents (grouping, registry)
│   │   │   │   └── resolution/          # Resolution agents (same_type, cross_type)
│   │   │   ├── transcription/           # Transcriber interface and provider implementations
│   │   │   ├── pipeline/                # Orchestration
│   │   │   │   ├── extraction/          # Extraction sub-pipeline (identification, scoring, resolution)
│   │   │   │   └── ingestion/           # Ingestion sub-pipeline (audio/text validation, blob upload)
│   │   │   ├── repositories/            # NodeRepository and ops/blob repository facades
│   │   │   ├── services/                # Domain services (GraphService, JobService, AdminService, …)
│   │   │   └── platform/                # Deployment-layer concerns
│   │   │       ├── api/                 # FastAPI routers, auth, app state, startup
│   │   │       ├── worker/              # In-process async task queue and job worker
│   │   │       └── observability/       # MLflow tracing, usage tracking, latency metrics
│   │   ├── eval/                        # Eval harnesses and calibration meta-scorers (tooling, not runtime)
│   │   └── cli/                         # CLI entry points (seshat api, seshat eval, seshat migrate)
│   └── seshat_ui/                       # Streamlit web UI (separate package; thin client over the API)
├── scripts/                             # Standalone helper scripts (not part of the package)
├── tests/
├── data/
│   ├── eval/                            # Ground-truth YAML corpus fixtures for eval harnesses
│   └── fixtures/                        # Generated fixtures (e.g. synthetic audio)
├── alembic/                             # DB migration scripts
├── docs/                                # Guides, architecture docs, SDD, design specs
├── docker/                              # Dockerfiles for the API and UI images, plus LocalStack init
├── docker-compose.yml                   # The full local stack
└── pyproject.toml                       # Single source of truth for deps, tool config, metadata
```

## Documentation map

**Guides — how to use it:**

- `docs/user-guide.md` → End-to-end walkthrough of the web UI, following a worked scenario.
- `docs/eval-user-guide.md` → Running the evaluation harnesses, reading results in MLflow, and calibrating thresholds.
- `docs/configuration.md` → Full reference of every config parameter and its environment variable.

**Design & architecture — how it works:**

- `docs/primer.md` → Developer primer: narrative overview and end-to-end job walkthrough.
- `docs/architecture.md` → Architecture summary: key design decisions and rationale.
- `docs/seshat-sdd.md` → Solution Design Document: implementation-oriented system design.
- `docs/superpowers/specs/2026-04-21-seshat-design.md` → Full design spec and detailed contracts.
