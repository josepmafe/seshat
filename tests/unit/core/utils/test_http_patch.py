import ssl

import pytest

from seshat.core.utils import http_patch


@pytest.fixture(autouse=True)
def _restore_ssl_context():
    """truststore.inject_into_ssl() swaps ssl.SSLContext globally; restore it per test."""
    original = ssl.SSLContext
    http_patch._truststore_injected = False
    try:
        yield
    finally:
        ssl.SSLContext = original
        http_patch._truststore_injected = False


class TestInjectOsTruststore:
    def test_replaces_ssl_context_with_truststore(self):
        before = ssl.SSLContext
        http_patch.inject_os_truststore()
        assert ssl.SSLContext is not before  # truststore installed its own SSLContext

    def test_new_ssl_context_still_verifies(self):
        # truststore keeps verification ON (unlike the old disable-verify hack); it only
        # changes WHERE trust comes from (the OS store).
        http_patch.inject_os_truststore()
        ctx = ssl.create_default_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_is_idempotent(self):
        http_patch.inject_os_truststore()
        after_first = ssl.SSLContext
        http_patch.inject_os_truststore()
        assert ssl.SSLContext is after_first  # second call does not re-inject
