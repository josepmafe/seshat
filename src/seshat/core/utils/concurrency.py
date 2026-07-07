from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, TypeVar

import anyio

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")


async def run_in_thread(fn: Callable[..., _T], *args: Any, cancellable: bool = False) -> _T:
    """Run a blocking synchronous function in a worker thread.

    Uses anyio.to_thread.run_sync for cancellation propagation support.
    Set cancellable=True to allow the thread to be abandoned on task cancellation
    (only safe if fn has no irreversible side-effects mid-run).
    """
    wrapped = functools.partial(fn, *args) if args else fn
    return await anyio.to_thread.run_sync(wrapped, abandon_on_cancel=cancellable)
