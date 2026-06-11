from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.config.settings import SecretsConfig


logger = get_logger(__name__)


class AbstractSecretsResolver(ABC):
    def __init__(self, config: SecretsConfig) -> None:
        self._config = config
        # Cache avoids repeated calls to the secrets backend. No TTL: rotated secrets require a process restart.
        self._cache: dict[str, str] = {}

    def get_secret(self, key: str) -> str:
        if key not in self._cache:
            logger.info("Fetching secret for %r key", key)
            self._cache[key] = self._fetch_secret(key)
        else:
            logger.debug("Using cached secret for %r key", key)
        return self._cache[key]

    @abstractmethod
    def _fetch_secret(self, key: str) -> str: ...
