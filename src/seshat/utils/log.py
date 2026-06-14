import logging
from contextvars import ContextVar

_job_id_var: ContextVar[str] = ContextVar("job_id", default="")
_task_num_var: ContextVar[str] = ContextVar("task_num", default="")

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(job_id)s%(task_num)s] %(name)s: %(message)s"
_NOISY_LOGGERS = (
    "aiobotocore",
    "botocore",
    "httpx",
    "langchain",
    "langchain_core",
    "langchain_aws",
    "langchain_openai",
    "mlflow",
)


def set_job_id(job_id: str) -> None:
    _job_id_var.set(job_id)


def set_task_num(num: int) -> None:
    # Stored as "#N" so it appends naturally after job_id; empty string when not set keeps
    # the bracket readable as "[job-id]" in production (no dangling separator).
    _task_num_var.set(f"#{num}")


class _JobIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id_var.get()  # type: ignore[attr-defined]
        record.task_num = _task_num_var.get()  # type: ignore[attr-defined]
        return True


_job_id_filter = _JobIdFilter()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addFilter(_job_id_filter)
    return logger


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a StreamHandler with job_id in the format. Call once at app startup."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_job_id_filter)
    logging.root.addHandler(handler)
    logging.root.setLevel(level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
