"""Local development entrypoint for the Seshat API.

Sets WindowsSelectorEventLoopPolicy before uvicorn starts (Windows requirement),
and patches httpx.Client to disable SSL verification for corporate proxy environments.
Both patches must be applied before any seshat imports.
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Corporate SSL proxy does certificate inspection — patch httpx before any
# library (assemblyai, etc.) imports and caches its own httpx.Client instance.
import httpx

_orig_httpx_init = httpx.Client.__init__


def _httpx_no_verify(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    _orig_httpx_init(self, *args, **kwargs)


httpx.Client.__init__ = _httpx_no_verify

import uvicorn  # noqa: E402

from seshat.api.app import create_app  # noqa: E402

if __name__ == "__main__":
    from seshat.utils.log import configure_logging

    configure_logging()

    async def _serve() -> None:
        # Docker uses port 8000 for the API by default, so we use 8001 so we can run both APIs without port conflicts.
        config = uvicorn.Config(create_app(), host="0.0.0.0", port=8001, log_config=None)
        await uvicorn.Server(config).serve()

    asyncio.run(_serve())
