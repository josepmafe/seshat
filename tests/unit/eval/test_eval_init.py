from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from seshat.eval import require_eval_deps


def test_require_eval_deps_does_not_raise_when_rapidfuzz_available():
    # rapidfuzz is installed in this project — no exception should be raised
    require_eval_deps()


def test_require_eval_deps_raises_importerror_when_rapidfuzz_missing():
    with (
        patch.dict(sys.modules, {"rapidfuzz": None}),
        pytest.raises(ImportError, match=r"seshat\.eval package requires"),
    ):
        require_eval_deps()
