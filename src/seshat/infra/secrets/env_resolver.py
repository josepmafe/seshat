from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from seshat.infra.secrets.base_resolver import AbstractSecretsResolver

if TYPE_CHECKING:
    from seshat.core.config.settings import SecretsConfig


class EnvSecretsResolver(AbstractSecretsResolver):
    def __init__(self, config: SecretsConfig) -> None:
        super().__init__(config)
        load_dotenv(override=False)

    def _fetch_secret(self, key: str) -> str:
        value = os.environ.get(key.upper())
        if value is None:
            raise KeyError(key)
        if not value:
            raise ValueError(f"Secret {key!r} is set but empty — check your environment")
        return value
