from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seshat.core.config.settings import LoggingConfig

_job_id_var: ContextVar[str] = ContextVar("job_id", default="")
_task_num_var: ContextVar[str] = ContextVar("task_num", default="")

_LOG_FORMAT = "%(asctime)s %(levelname)s%(job_ctx)s %(name)s: %(message)s"


def set_job_id(job_id: str) -> None:
    _job_id_var.set(job_id)


def set_task_num(num: int) -> None:
    # Stored as "#N" so it appends naturally after job_id; empty string when not set keeps
    # the bracket readable as "[job-id]" in production (no dangling separator).
    _task_num_var.set(f"#{num}")


class _JobIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        job_id = _job_id_var.get()
        task_num = _task_num_var.get()
        # Omit brackets entirely when there is no job context.
        record.job_ctx = f" [{job_id}{task_num}]" if job_id else ""  # type: ignore[attr-defined]
        return True


_job_id_filter = _JobIdFilter()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addFilter(_job_id_filter)
    return logger


def configure_logging(config: LoggingConfig | None = None) -> None:
    """Configure a StreamHandler with job_id in the format. Call once at app startup."""
    from seshat.core.config.settings import LoggingConfig as _LoggingConfig  # avoid circular import at module load

    if config is None:
        config = _LoggingConfig()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_job_id_filter)
    logging.root.addHandler(handler)
    logging.root.setLevel(config.level)

    for name, level in config.noisy_loggers.items():
        logging.getLogger(name).setLevel(level)
