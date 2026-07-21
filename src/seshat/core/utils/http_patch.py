"""Shared HTTP client tweaks.

Home of the OS-truststore injection used when running behind a corporate proxy
whose CA is not in Python's bundled certifi bundle but IS in the OS trust store
(e.g. a Windows laptop where IT provisions the proxy's CA into Windows, but
Python ships its own cert bundle). Gated by config; off by default.
"""

from __future__ import annotations

import truststore

from seshat.core.utils.log import get_logger

logger = get_logger(__name__)

_truststore_injected = False


def inject_os_truststore() -> None:
    """Make ssl.SSLContext source trust from the OS certificate store.

    Verification stays ON — this only changes WHERE the trust anchors come from
    (OS store instead of Python's bundled certifi CAs), fixing
    'CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate' on a
    corporate network whose proxy CA is provisioned into the OS but not seen by
    Python. Affects every consumer of ssl.SSLContext (httpx sync + async,
    boto3, aiohttp, requests, ...), since it patches the ssl module itself,
    not any one HTTP client. Idempotent — calling twice does not re-inject.
    """
    global _truststore_injected
    if _truststore_injected:
        return

    truststore.inject_into_ssl()
    _truststore_injected = True
    logger.warning("ssl.SSLContext now sources trust from the OS certificate store (config opt-in).")
