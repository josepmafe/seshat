import pytest


def assert_invalid_scheme_raises(cls) -> None:
    with pytest.raises(ValueError, match="Invalid connection string"):
        cls._validate_connection_string("mysql://user:pass@host/db")


def assert_credentials_not_in_error(cls) -> None:
    with pytest.raises(ValueError, match="Invalid connection string") as exc_info:
        cls._validate_connection_string("mysql://secret:hunter2@host/db")
    assert "secret" not in str(exc_info.value)
    assert "hunter2" not in str(exc_info.value)
