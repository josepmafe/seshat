from __future__ import annotations

import bcrypt

from seshat.core.utils.concurrency import run_in_thread
from seshat.core.utils.log import get_logger

logger = get_logger(__name__)


class AuthenticationError(Exception):
    pass


async def verify_api_key(
    key: str,
    stored_keys: list[tuple[str, str, str]],
) -> tuple[str, str]:
    """Check key against (hash, user_id, role) tuples using constant-time bcrypt.

    Returns (user_id, role) on success.
    """
    for key_hash, user_id, role in stored_keys:
        try:
            match = await run_in_thread(bcrypt.checkpw, key.encode(), key_hash.encode())
        except ValueError:
            logger.warning("Skipping api_keys row for user %r: stored hash is not a valid bcrypt hash", user_id)
            continue

        if match:
            return user_id, role
    raise AuthenticationError("Invalid API key")
