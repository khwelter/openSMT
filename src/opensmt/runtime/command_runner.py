from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CommandJob:
    job_id: str
    name: str
    state: str  # queued | running | succeeded | failed | canceled
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class CommandRunner:
    """Tracks asynchronous command execution and outcomes."""

    def __init__(self, max_history: int = 400) -> None:
        self._max_history = max(50, int(max_history))
        self._jobs: dict[str, CommandJob] = {}
        self._order: deque[str] = deque()
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def submit(self, name: str, command: Callable[[], Awaitable[None]]) -> str:
        """Queue a command and return a stable job id immediately."""
        job_id = uuid.uuid4().hex
        job = CommandJob(
            job_id=job_id,
            name=name,
            state="queued",
            created_at=time.time(),
        )
        self._jobs[job_id] = job
        self._order.append(job_id)
        self._trim_history()

        task = asyncio.create_task(
            self._run(job_id, command),
            name=f"cmd-{name}-{job_id[:8]}",
        )
        self._tasks[job_id] = task
        return job_id

    def get(self, job_id: str) -> dict[str, float | str | None] | None:
        job = self._jobs.get(job_id)
        return None if job is None else job.to_dict()

    def recent(self, limit: int = 50) -> list[dict[str, float | str | None]]:
        n = max(1, min(int(limit), self._max_history))
        ids = list(self._order)[-n:]
        ids.reverse()  # newest first
        return [self._jobs[job_id].to_dict() for job_id in ids if job_id in self._jobs]

    def cancel(self, job_id: str) -> str:
        """Cancel a job if possible.

        Returns one of: unknown, already_done, cancel_requested.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return "unknown"
        if job.state in {"succeeded", "failed", "canceled"}:
            return "already_done"

        task = self._tasks.get(job_id)
        if task is None:
            return "already_done"
        task.cancel()
        return "cancel_requested"

    async def _run(self, job_id: str, command: Callable[[], Awaitable[None]]) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return

        job.state = "running"
        job.started_at = time.time()
        try:
            await command()
            job.state = "succeeded"
        except asyncio.CancelledError:
            job.state = "canceled"
            job.error = "canceled"
            raise
        except Exception as exc:
            job.state = "failed"
            job.error = str(exc)
            log.exception("Command job %s (%s) failed", job.job_id, job.name)
        finally:
            job.finished_at = time.time()
            self._tasks.pop(job_id, None)

    def _trim_history(self) -> None:
        while len(self._order) > self._max_history:
            oldest = self._order.popleft()
            self._jobs.pop(oldest, None)
