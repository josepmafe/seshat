from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.ops_store.pg_store import PostgresOpsStore
from seshat.secrets.factory import get_secrets_resolver

if TYPE_CHECKING:
    from seshat.core.config.settings import SeshatConfig


def get_ops_store(seshat_config: SeshatConfig) -> PostgresOpsStore:
    secrets = get_secrets_resolver(seshat_config)
    connection_string = secrets.get_secret(seshat_config.ops_store.connection_secret_key)
    return PostgresOpsStore(seshat_config.ops_store, connection_string)
