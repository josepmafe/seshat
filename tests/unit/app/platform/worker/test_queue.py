import asyncio

from seshat.app.platform.worker.queue import AsyncioTaskQueue
from seshat.core.models.enums import JobStatus


class TestAsyncioTaskQueue:
    async def test_enqueue_and_get_status(self):
        queue = AsyncioTaskQueue()

        async def noop():
            pass

        await queue.enqueue("job-1", noop)
        await asyncio.sleep(0)
        status = await queue.get_status("job-1")
        assert status == JobStatus.DONE

    async def test_get_status_unknown_job(self):
        queue = AsyncioTaskQueue()
        status = await queue.get_status("nonexistent-job")
        assert status is None

    async def test_cancel_pending_job(self):
        queue = AsyncioTaskQueue()
        gate = asyncio.Event()

        async def slow():
            await gate.wait()

        await queue.enqueue("job-2", slow)
        cancelled = await queue.cancel("job-2")
        gate.set()
        assert cancelled is True

    async def test_cancel_completed_job_returns_false(self):
        queue = AsyncioTaskQueue()

        async def noop():
            pass

        await queue.enqueue("job-done", noop)
        await asyncio.sleep(0)
        cancelled = await queue.cancel("job-done")
        assert cancelled is False

    async def test_cancel_nonexistent_job_returns_false(self):
        queue = AsyncioTaskQueue()
        assert await queue.cancel("no-such-job") is False

    async def test_cancelled_task_sets_failed_status(self):
        queue = AsyncioTaskQueue()
        gate = asyncio.Event()

        async def blocker():
            await gate.wait()

        await queue.enqueue("job-cancel", blocker)
        await asyncio.sleep(0)  # let the task start and reach gate.wait()
        await queue.cancel("job-cancel")
        await asyncio.sleep(0)  # let cancellation propagate through _run
        status = await queue.get_status("job-cancel")
        assert status == JobStatus.FAILED
