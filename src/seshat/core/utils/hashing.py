from __future__ import annotations

import hashlib


def fingerprint(text: str) -> str:
    """Return an 8-char hex digest of `text`, used as a cache-key component."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]
