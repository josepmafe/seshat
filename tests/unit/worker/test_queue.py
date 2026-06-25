import asyncio

from seshat.models.enums import JobStatus
from seshat.worker.queue import AsyncioTaskQueue


class TestAsyncioTaskQueue:
    async def test_enqueue_and_get_status(self):
        queue = AsyncioTaskQueue()

        async def noop():
            pass

        job_id = await queue.enqueue(noop)
        assert isinstance(job_id, str)
        status = await queue.get_status(job_id)
        assert status in (JobStatus.PENDING, JobStatus.DONE)

    async def test_get_status_unknown_job(self):
        queue = AsyncioTaskQueue()
        status = await queue.get_status("nonexistent-job")
        assert status is None

    async def test_cancel_pending_job(self):
        queue = AsyncioTaskQueue()
        gate = asyncio.Event()

        async def slow():
            await gate.wait()

        job_id = await queue.enqueue(slow)
        cancelled = await queue.cancel(job_id)
        gate.set()
        assert cancelled is True
