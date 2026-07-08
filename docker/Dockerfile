FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS deps
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev


FROM deps AS migrate
COPY alembic.ini .
COPY alembic/ alembic/
COPY src/ src/
CMD ["uv", "run", "--no-sync", "alembic", "upgrade", "head"]


FROM deps AS api
COPY src/ src/
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "seshat.app.platform.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
