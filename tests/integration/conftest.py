import asyncio
import os
import socket
import sys

import pytest
from dotenv import load_dotenv

load_dotenv()

_LOCALSTACK_PORT = int(os.environ.get("LOCALSTACK_PORT", 4566))
LOCALSTACK_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
LOCALSTACK_TEST_BUCKET = "seshat-test"

_PG_USER = os.environ.get("POSTGRES_USER", "seshat")
_PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "seshat")
_PG_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
_PG_DB = os.environ.get("POSTGRES_DB", "seshat")
_PG_ADMIN_URL = f"postgresql://{_PG_USER}:{_PG_PASSWORD}@localhost:{_PG_PORT}/{_PG_DB}"
_PG_TEST_DB = "seshat_test"
_PG_TEST_URL = f"postgresql://{_PG_USER}:{_PG_PASSWORD}@localhost:{_PG_PORT}/{_PG_TEST_DB}"

# DDL mirroring alembic/versions/001_initial_ops_schema.py.
# TODO(task10): replace with `alembic upgrade head` once migrations are in place.
_OPS_DDL = """
CREATE SCHEMA IF NOT EXISTS ops;
CREATE TABLE IF NOT EXISTS ops.kb_nodes (
    node_id         TEXT PRIMARY KEY,
    schema_version  TEXT NOT NULL DEFAULT '1.0',
    job_id          TEXT NOT NULL,
    type            TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    confidence      FLOAT NOT NULL,
    source_quote    TEXT NOT NULL,
    status          TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'current',
    metadata        JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS ops.kb_relationships (
    source_id   TEXT NOT NULL REFERENCES ops.kb_nodes(node_id),
    target_id   TEXT NOT NULL REFERENCES ops.kb_nodes(node_id),
    rel_type    TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (source_id, target_id, rel_type)
);
CREATE INDEX IF NOT EXISTS ix_kb_relationships_target_id
    ON ops.kb_relationships (target_id);
"""


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _openai_reachable() -> bool:
    import httpx

    try:
        response = httpx.get("https://api.openai.com/v1/models", timeout=5)
        return response.status_code == 200
    except httpx.RequestError:
        return False


SKIP_IF_NO_POSTGRES = pytest.mark.skipif(
    not _port_open("localhost", _PG_PORT),
    reason="Postgres not reachable — run: docker compose up -d postgres",
)


SKIP_IF_NO_LOCALSTACK = pytest.mark.skipif(
    not _port_open("localhost", _LOCALSTACK_PORT),
    reason="LocalStack not reachable — run: docker compose up -d localstack",
)

SKIP_IF_NO_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not _openai_reachable(),
    reason="OpenAI API not reachable or OPENAI_API_KEY not set",
)


@pytest.fixture(scope="session")
def event_loop_policy():
    # psycopg async requires SelectorEventLoop; Windows defaults to ProactorEventLoop
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
async def pg_test_url():
    """Create a throw-away seshat_test database, yield its URL, then drop it.

    Keeps integration tests isolated from real data in the seshat database.
    Skipped automatically when Postgres is not reachable (same check as SKIP_IF_NO_POSTGRES).
    """
    if not _port_open("localhost", _PG_PORT):
        pytest.skip("Postgres not reachable — run: docker compose up -d postgres")

    import asyncpg

    admin = await asyncpg.connect(_PG_ADMIN_URL)
    await admin.execute(f"DROP DATABASE IF EXISTS {_PG_TEST_DB}")
    await admin.execute(f"CREATE DATABASE {_PG_TEST_DB} OWNER {_PG_USER}")
    await admin.close()

    test_url = _PG_TEST_URL
    conn = await asyncpg.connect(test_url)
    await conn.execute(_OPS_DDL)
    await conn.close()

    yield test_url

    admin = await asyncpg.connect(_PG_ADMIN_URL)
    await admin.execute(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='{_PG_TEST_DB}'")
    await admin.execute(f"DROP DATABASE IF EXISTS {_PG_TEST_DB}")
    await admin.close()


@pytest.fixture(scope="session")
async def localstack_secretsmanager_url():
    """Create a throw-away Secrets Manager in LocalStack, yield the endpoint URL, then delete it.

    Keeps secrets-manager integration tests isolated from the dev secrets.
    Skipped automatically when LocalStack is not reachable.
    """
    return _get_localstack_url()


@pytest.fixture(scope="session")
async def localstack_s3_url():
    """Create a throw-away S3 bucket in LocalStack, yield the endpoint URL, then delete it.

    Keeps blob-store integration tests isolated from the dev bucket.
    Skipped automatically when LocalStack is not reachable.
    """
    endpoint = _get_localstack_url()

    import aioboto3

    session = aioboto3.Session()
    async with session.client("s3", region_name=LOCALSTACK_REGION, endpoint_url=endpoint) as s3:
        await s3.create_bucket(
            Bucket=LOCALSTACK_TEST_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": LOCALSTACK_REGION},
        )

    yield endpoint

    async with session.client("s3", region_name=LOCALSTACK_REGION, endpoint_url=endpoint) as s3:
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=LOCALSTACK_TEST_BUCKET):
            for obj in page.get("Contents", []):
                await s3.delete_object(Bucket=LOCALSTACK_TEST_BUCKET, Key=obj["Key"])
        await s3.delete_bucket(Bucket=LOCALSTACK_TEST_BUCKET)


def _get_localstack_url():
    if not _port_open("localhost", _LOCALSTACK_PORT):
        pytest.skip("LocalStack not reachable — run: docker compose up -d localstack")
    return f"http://localhost:{_LOCALSTACK_PORT}"
