import os
import socket

import pytest
from dotenv import load_dotenv

load_dotenv()

_LOCALSTACK_PORT = int(os.environ.get("LOCALSTACK_PORT", 4566))


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


SKIP_IF_NO_LOCALSTACK = pytest.mark.skipif(
    not _port_open("localhost", LOCALSTACK_PORT),
    reason="LocalStack not reachable — run: docker compose up -d localstack",
)


@pytest.fixture(scope="session")
async def localstack_secretsmanager_url():
    """Create a throw-away Secrets Manager in LocalStack, yield the endpoint URL, then delete it.

    Keeps secrets-manager integration tests isolated from the dev secrets.
    Skipped automatically when LocalStack is not reachable.
    """
    return _get_localstack_url()


def _get_localstack_url():
    if not _port_open("localhost", _LOCALSTACK_PORT):
        pytest.skip("LocalStack not reachable — run: docker compose up -d localstack")
    return f"http://localhost:{_LOCALSTACK_PORT}"
