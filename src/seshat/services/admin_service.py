from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import bcrypt

from seshat.core.utils.concurrency import run_in_thread
from seshat.infra.repositories.ops_repository import ApiKeyAlreadyRevokedError, ApiKeyNotFoundError

if TYPE_CHECKING:
    from seshat.core.models.enums import UserRole
    from seshat.infra.repositories.ops_repository import OpsRepository

__all__ = ["AdminService", "ApiKeyAlreadyRevokedError", "ApiKeyNotFoundError"]


class AdminService:
    def __init__(self, ops_repo: OpsRepository) -> None:
        self._ops = ops_repo

    async def create_api_key(self, user_id: str, role: UserRole) -> tuple[str, str, UserRole]:
        """Return (plaintext_key, user_id, role)."""
        plaintext = secrets.token_urlsafe(32)
        key_hash = await run_in_thread(bcrypt.hashpw, plaintext.encode(), bcrypt.gensalt())
        await self._ops.create_api_key(key_hash.decode(), user_id, role, datetime.now(UTC))
        return plaintext, user_id, role

    async def get_api_keys(self) -> list[tuple[str, str, str]]:
        return await self._ops.get_api_keys()

    async def list_api_keys(self) -> list[Any]:
        return await self._ops.list_api_keys()

    async def revoke_api_key(self, key_id: int) -> None:
        await self._ops.revoke_api_key(key_id, datetime.now(UTC))
