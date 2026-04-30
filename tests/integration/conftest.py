import os
import socket

import pytest

POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
LOCALSTACK_PORT = int(os.environ.get("LOCALSTACK_PORT", 4566))


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


SKIP_IF_NO_POSTGRES = pytest.mark.skipif(
    not _port_open("localhost", POSTGRES_PORT),
    reason="Postgres not reachable — run: docker compose up -d postgres",
)

SKIP_IF_NO_LOCALSTACK = pytest.mark.skipif(
    not _port_open("localhost", LOCALSTACK_PORT),
    reason="LocalStack not reachable — run: docker compose up -d localstack",
)

SKIP_IF_NO_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
