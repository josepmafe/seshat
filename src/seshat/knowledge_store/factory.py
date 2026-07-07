from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.core.utils.log import get_logger
from seshat.knowledge_store.pg_store import PostgresKBStore
from seshat.secrets.factory import get_secrets_resolver

if TYPE_CHECKING:
    from seshat.core.config.settings import SeshatConfig

logger = get_logger(__name__)


def get_kb_store(config: SeshatConfig) -> PostgresKBStore:
    secrets = get_secrets_resolver(config)
    connection_string = secrets.get_secret(config.kb_store.connection_secret_key)

    logger.debug("Initialising KB store (schema=%s)", config.kb_store.schema_name)
    return PostgresKBStore(config.kb_store, connection_string)
