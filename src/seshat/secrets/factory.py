from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from seshat.core.models.enums import SecretsProvider
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.core.config.settings import SecretsConfig, SeshatConfig
    from seshat.secrets.base_resolver import AbstractSecretsResolver


logger = get_logger(__name__)


def get_secrets_resolver(config: SeshatConfig) -> AbstractSecretsResolver:
    return _cached_resolver(config.secrets)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def _cached_resolver(config: SecretsConfig) -> AbstractSecretsResolver:
    # Split from get_secrets_resolver so lru_cache operates on the hashable SecretsConfig,
    # not the full SeshatConfig which is unhashable.
    logger.debug("Initialising secrets resolver: %s", config.provider)
    match config.provider:
        case SecretsProvider.ENV:
            from seshat.secrets.env_resolver import EnvSecretsResolver

            return EnvSecretsResolver(config)
        case SecretsProvider.AWS:
            from seshat.secrets.aws_resolver import AWSSecretsResolver

            return AWSSecretsResolver(config)
        case _:
            raise ValueError(f"Unknown secrets provider: {config.provider!r}")
